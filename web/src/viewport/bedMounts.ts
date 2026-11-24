import * as THREE from "three";
import { BedMountType, ViewMode } from "../types";

export const BED_MOUNT_STANDOFF_HEIGHT_MM: Record<BedMountType, number> = {
  trident_3pt_standoffs: 10,
  floating_4pt: 12,
  whopping_pochard_kinematic: 14,
  rigid_4pt: 14,
  kinematic_3pt: 16,
  silicone_4pt: 16,
  springs_4pt: 20,
};

export interface BedMountsBuildParams {
  bedMountType: BedMountType;
  supportPtsThree: THREE.Vector3[];
  zArmEndPtsThree?: THREE.Vector3[];
  bedBaseCenter: THREE.Vector3;
  bedBottomY: number;
  bedDispDelta: THREE.Vector3;
  scale: number;
  viewMode: ViewMode;
  toolheadPunchIn?: boolean;
  useSensBedTint?: boolean;
  bedColor?: THREE.Color;
  bedEmissive?: THREE.Color;
  bedEmissiveIntensity?: number;
  onTagMesh?: (mesh: THREE.Mesh) => void;
}

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
 * Create a continuous helical coil spring geometry (flat-ground ends) along -Y.
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
    const y = yTop - yNorm * span;
    pts.push(new THREE.Vector3(Math.cos(theta) * helixRadius, y, Math.sin(theta) * helixRadius));
  }
  const curve = new THREE.CatmullRomCurve3(pts);
  return new THREE.TubeGeometry(curve, 84, wireRadius, 8, false);
}

/**
 * Create an extruded ribbed/knurled leveling thumbwheel centered at origin in XZ plane.
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
 * Create a realistic barreled & subtly ribbed 70A silicone bed isolator column along -Y.
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
 * Create a continuous S-stepped 301 spring-steel leaf flexure blade profile in XY extruded along Z.
 */
function createLeafFlexureBladeGeometry(
  s: number,
  bladeWidth: number,
  bladeThick: number
): THREE.ExtrudeGeometry {
  const halfT = bladeThick * 0.5;
  // Local +X = radial inward (bed-side upper clamp), local -X = radial outward (subframe lower block)
  const topPts: Array<[number, number]> = [
    [12.2 * s, -5.0 * s + halfT],
    [3.0 * s, -5.0 * s + halfT],
    [-0.5 * s, -6.5 * s + halfT],
    [-4.2 * s, -8.2 * s + halfT],
    [-13.0 * s, -8.2 * s + halfT],
  ];
  const botPts: Array<[number, number]> = [
    [-13.0 * s, -8.2 * s - halfT],
    [-4.6 * s, -8.2 * s - halfT],
    [-0.9 * s, -6.5 * s - halfT],
    [2.6 * s, -5.0 * s - halfT],
    [12.2 * s, -5.0 * s - halfT],
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
 * Build realistic engineering-accurate 3D CAD bed mount assemblies for all 7 `BedMountType` variants.
 * All geometries and materials are attached to meshes inside the returned Group so `disposeGroupChildren`
 * frees every WebGL resource cleanly on scene rebuild.
 */
export function buildBedMountsGroup(params: BedMountsBuildParams): THREE.Group {
  const {
    bedMountType,
    supportPtsThree,
    zArmEndPtsThree,
    bedBaseCenter,
    bedBottomY,
    bedDispDelta,
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

  if (supportPtsThree.length === 0) {
    return root;
  }

  const isModal = viewMode === "modal";
  const isTransparent = toolheadPunchIn || isModal;
  const opacity = toolheadPunchIn ? 0.14 : isModal ? 0.68 : 1.0;

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
      transparent: isTransparent,
      opacity,
      depthWrite: !toolheadPunchIn,
    });
  };

  const tag = (mesh: THREE.Mesh): THREE.Mesh => {
    if (onTagMesh) onTagMesh(mesh);
    return mesh;
  };

  // Place each mount assembly directly under `bedBottomY` (shifted by `bedDispDelta` in FEA mode).
  const placeMountGroup = (grp: THREE.Group, sp: THREE.Vector3): void => {
    grp.position.set(
      sp.x + bedDispDelta.x,
      bedBottomY + bedDispDelta.y,
      sp.z + bedDispDelta.z
    );
    root.add(grp);
  };

  if (bedMountType === "trident_3pt_standoffs") {
    // Stock Voron / Siboor Trident 3-Pt Phenolic Bakelite Thermal-Isolation Standoffs (10 mm) + GE5C
    const bakeliteMat = makeMat({
      color: 0x7c2d12, // phenolic reddish-brown / dark amber Bakelite
      metalness: 0.08,
      roughness: 0.42,
      sensTint: true,
    });
    const stainlessMat = makeMat({
      color: 0xdce4ee,
      metalness: 0.88,
      roughness: 0.22,
    });
    const socketDarkMat = makeMat({
      color: 0x1e293b,
      metalness: 0.55,
      roughness: 0.5,
    });

    const flangeGeo = new THREE.CylinderGeometry(8.4 * s, 8.4 * s, 2.0 * s, 18);
    const barrelGeo = new THREE.CylinderGeometry(6.6 * s, 6.6 * s, 6.0 * s, 18);
    const fenderWasherGeo = new THREE.CylinderGeometry(6.2 * s, 6.2 * s, 1.1 * s, 16);
    const screwShankGeo = new THREE.CylinderGeometry(2.0 * s, 2.0 * s, 12.6 * s, 10);
    const screwHeadGeo = new THREE.CylinderGeometry(3.6 * s, 3.6 * s, 3.2 * s, 14);
    const hexRecessGeo = new THREE.CylinderGeometry(1.7 * s, 1.7 * s, 0.4 * s, 6);

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
    const fixedCollarGeo = new THREE.CylinderGeometry(9.6 * s, 9.6 * s, 2.4 * s, 20);
    const fixedCollarMat = makeMat({
      color: 0xe8eef5,
      metalness: 0.92,
      roughness: 0.16,
    });

    const slottedWasherGeo = hasSlottedFront
      ? new THREE.BoxGeometry(20.0 * s, 1.7 * s, 11.4 * s)
      : null;
    const slotRecessGeo = hasSlottedFront
      ? new THREE.BoxGeometry(8.8 * s, 1.9 * s, 3.0 * s)
      : null;
    const slottedWasherMat = hasSlottedFront
      ? makeMat({
          color: 0x38bdf8, // subtle cyan-steel anodized slotted thermal-relief washer
          metalness: 0.68,
          roughness: 0.28,
        })
      : null;

    supportPtsThree.forEach((sp, idx) => {
      const grp = new THREE.Group();
      const isRearFixed = idx === rearFixedIdx;
      grp.rotation.y = radialYawToCenter(sp, bedBaseCenter);

      // Top & bottom insulating Bakelite flanges + central stepped phenolic column (10 mm total)
      const topFlange = tag(new THREE.Mesh(flangeGeo, bakeliteMat));
      topFlange.position.y = -1.0 * s;
      grp.add(topFlange);

      const barrel = tag(new THREE.Mesh(barrelGeo, bakeliteMat));
      barrel.position.y = -5.0 * s;
      grp.add(barrel);

      const botFlange = tag(new THREE.Mesh(flangeGeo, bakeliteMat));
      botFlange.position.y = -9.0 * s;
      grp.add(botFlange);

      // Visual cue: solid machined stainless locator collar on rear-center fixed mount vs
      // radially oriented cyan/steel slotted thermal-relief washer on the two front mounts
      if (isRearFixed) {
        const fixedCollar = tag(new THREE.Mesh(fixedCollarGeo, fixedCollarMat));
        fixedCollar.position.y = -1.3 * s;
        grp.add(fixedCollar);
      } else if (slottedWasherGeo && slotRecessGeo && slottedWasherMat) {
        const slotWasher = tag(new THREE.Mesh(slottedWasherGeo, slottedWasherMat));
        slotWasher.position.y = -10.85 * s;
        grp.add(slotWasher);

        const slotRecess = new THREE.Mesh(slotRecessGeo, socketDarkMat);
        slotRecess.position.set(1.2 * s, -10.85 * s, 0);
        grp.add(slotRecess);
      }

      // Stainless steel M4 fender washers & socket-head cap screw running through spacer
      const topFender = new THREE.Mesh(fenderWasherGeo, stainlessMat);
      topFender.position.y = -0.55 * s;
      grp.add(topFender);

      const botFender = new THREE.Mesh(fenderWasherGeo, stainlessMat);
      botFender.position.y = isRearFixed ? -10.55 * s : -12.1 * s;
      grp.add(botFender);

      const shank = new THREE.Mesh(screwShankGeo, stainlessMat);
      shank.position.y = -6.3 * s;
      grp.add(shank);

      const headY = isRearFixed ? -12.7 * s : -14.2 * s;
      const screwHead = new THREE.Mesh(screwHeadGeo, stainlessMat);
      screwHead.position.y = headY;
      grp.add(screwHead);

      const hexSocket = new THREE.Mesh(hexRecessGeo, socketDarkMat);
      hexSocket.position.y = headY - 1.55 * s;
      grp.add(hexSocket);

      placeMountGroup(grp, sp);
    });

    // Compact GE5C spherical rod-end bearing rings where the 2020 T-subframe arms meet the Z-carriages
    if (zArmEndPtsThree && zArmEndPtsThree.length > 0) {
      const ge5cOuterMat = makeMat({
        color: 0x94a3b8,
        metalness: 0.86,
        roughness: 0.24,
      });
      const ge5cBronzeMat = makeMat({
        color: 0xc9933b, // bronze/brass spherical inner ball
        metalness: 0.82,
        roughness: 0.24,
      });
      const ge5cClevisMat = makeMat({
        color: 0x334155,
        metalness: 0.72,
        roughness: 0.36,
      });

      const outerRaceGeo = new THREE.CylinderGeometry(8.6 * s, 8.6 * s, 5.8 * s, 18);
      const innerBallGeo = new THREE.SphereGeometry(6.2 * s, 16, 12);
      const ge5cPinGeo = new THREE.CylinderGeometry(2.5 * s, 2.5 * s, 10.6 * s, 12);
      const clevisCollarGeo = new THREE.BoxGeometry(10.5 * s, 6.8 * s, 11.0 * s);

      for (const zPt of zArmEndPtsThree) {
        const toCenter = new THREE.Vector3(bedBaseCenter.x - zPt.x, 0, bedBaseCenter.z - zPt.z);
        const horizLen = toCenter.length();
        const inwardOffset = horizLen > 0.35 ? 25.0 * s : 0;
        if (horizLen > 1e-4) toCenter.divideScalar(horizLen);

        const bearingGrp = new THREE.Group();
        bearingGrp.rotation.y = radialYawToCenter(zPt, bedBaseCenter);

        const outerRace = tag(new THREE.Mesh(outerRaceGeo, ge5cOuterMat));
        bearingGrp.add(outerRace);

        const bronzeBall = tag(new THREE.Mesh(innerBallGeo, ge5cBronzeMat));
        bearingGrp.add(bronzeBall);

        const centerPin = new THREE.Mesh(ge5cPinGeo, stainlessMat);
        bearingGrp.add(centerPin);

        const clevisCollar = new THREE.Mesh(clevisCollarGeo, ge5cClevisMat);
        clevisCollar.position.set(10.2 * s, 0, 0);
        bearingGrp.add(clevisCollar);

        bearingGrp.position.set(
          zPt.x + toCenter.x * inwardOffset + bedDispDelta.x * 0.25,
          zPt.y + bedDispDelta.y * 0.25,
          zPt.z + toCenter.z * inwardOffset + bedDispDelta.z * 0.25
        );
        root.add(bearingGrp);
      }
    }
  } else if (bedMountType === "kinematic_3pt") {
    // 3-Point Maxwell Kinematic Ball-in-V-Groove (16 mm)
    const lowerPuckMat = makeMat({
      color: 0x334155, // dark anodized CNC aluminum lower puck
      metalness: 0.78,
      roughness: 0.3,
      sensTint: true,
    });
    const upperCupMat = makeMat({
      color: 0xd4dde8, // magnetic stainless steel upper cup
      metalness: 0.88,
      roughness: 0.22,
    });
    const chromeBallMat = makeMat({
      color: 0xf1f5f9, // hardened G10 chrome steel bearing ball
      metalness: 0.96,
      roughness: 0.08,
    });
    const hardenedPinMat = makeMat({
      color: 0xcbd5e1, // hardened steel cylindrical V-groove pins
      metalness: 0.92,
      roughness: 0.14,
    });
    const neoMagnetMat = makeMat({
      color: 0xbae6fd, // recessed nickel-plated neodymium preload magnet disc
      metalness: 0.9,
      roughness: 0.12,
    });

    const upperFlangeGeo = new THREE.CylinderGeometry(10.6 * s, 10.6 * s, 2.0 * s, 18);
    const upperCupGeo = new THREE.CylinderGeometry(8.2 * s, 6.2 * s, 2.6 * s, 18);
    const ballGeo = new THREE.SphereGeometry(5.2 * s, 18, 14);
    const vPinGeo = new THREE.CylinderGeometry(2.5 * s, 2.5 * s, 16.0 * s, 12);
    const magnetGeo = new THREE.CylinderGeometry(4.6 * s, 4.6 * s, 1.8 * s, 16);
    const lowerPuckGeo = new THREE.CylinderGeometry(11.6 * s, 12.6 * s, 4.2 * s, 20);
    const baseBoltGeo = new THREE.CylinderGeometry(1.7 * s, 1.7 * s, 1.8 * s, 10);

    for (const sp of supportPtsThree) {
      const grp = new THREE.Group();
      // Orient twin V-groove pins radially toward bed center (parallel to local +X expansion vector)
      grp.rotation.y = radialYawToCenter(sp, bedBaseCenter);

      // Upper magnetic stainless cup bolted to bed underside
      const upperFlange = tag(new THREE.Mesh(upperFlangeGeo, upperCupMat));
      upperFlange.position.y = -1.0 * s;
      grp.add(upperFlange);

      const upperCup = tag(new THREE.Mesh(upperCupGeo, upperCupMat));
      upperCup.position.y = -3.3 * s;
      grp.add(upperCup);

      // Hardened chrome steel bearing ball seated cleanly between the twin V-groove pins
      const ball = tag(new THREE.Mesh(ballGeo, chromeBallMat));
      ball.position.y = -7.6 * s;
      grp.add(ball);

      // Twin hardened steel cylindrical V-groove pins parallel to local +X (radial vector)
      for (const side of [-1, 1]) {
        const pin = tag(new THREE.Mesh(vPinGeo, hardenedPinMat));
        pin.rotation.z = Math.PI * 0.5;
        pin.position.set(0, -10.9 * s, side * 4.3 * s);
        grp.add(pin);
      }

      // Recessed nickel-plated neodymium preload magnet disc in lower puck center
      const magnet = new THREE.Mesh(magnetGeo, neoMagnetMat);
      magnet.position.y = -12.2 * s;
      grp.add(magnet);

      // Anodized lower CNC puck on the subframe + flank clamp screws
      const lowerPuck = tag(new THREE.Mesh(lowerPuckGeo, lowerPuckMat));
      lowerPuck.position.y = -14.0 * s;
      grp.add(lowerPuck);

      for (const side of [-1, 1]) {
        const bBolt = new THREE.Mesh(baseBoltGeo, upperCupMat);
        bBolt.position.set(0, -11.6 * s, side * 8.6 * s);
        grp.add(bBolt);
      }

      placeMountGroup(grp, sp);
    }
  } else if (bedMountType === "whopping_pochard_kinematic") {
    // Whopping Pochard 301 Spring-Steel Leaf-Spring Kinematic (14 mm, Tap-Rated)
    const leafBladeMat = makeMat({
      color: 0xe2e8f0, // 301 stainless spring-steel leaf flexure blade
      metalness: 0.93,
      roughness: 0.16,
      sensTint: true,
    });
    const upperClampMat = makeMat({
      color: 0x3b485c, // bed-side machined CNC clamp block
      metalness: 0.78,
      roughness: 0.3,
    });
    const lowerBlockMat = makeMat({
      color: 0x273244, // stepped dark-anodized CNC lower mount block
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

    const upperClampGeo = new THREE.BoxGeometry(13.0 * s, 4.6 * s, 16.0 * s);
    const upperShimGeo = new THREE.BoxGeometry(10.5 * s, 1.4 * s, 13.6 * s);
    const leafBladeGeo = createLeafFlexureBladeGeometry(s, 12.6 * s, 1.05 * s);
    const lowerBaseGeo = new THREE.BoxGeometry(15.0 * s, 5.2 * s, 16.6 * s);
    const lowerStepGeo = new THREE.BoxGeometry(9.0 * s, 1.8 * s, 13.6 * s);
    const m3BoltGeo = new THREE.CylinderGeometry(2.1 * s, 2.1 * s, 2.4 * s, 10);
    const alignPinGeo = new THREE.CylinderGeometry(1.9 * s, 1.9 * s, 4.8 * s, 12);

    for (const sp of supportPtsThree) {
      const grp = new THREE.Group();
      // Orient leaf flexure blade radially toward bed center (local +X = radial inward)
      grp.rotation.y = radialYawToCenter(sp, bedBaseCenter);

      // Upper bed-side machined clamp block + twin stainless M3 socket-head bolts + radial alignment pin
      const upperClamp = tag(new THREE.Mesh(upperClampGeo, upperClampMat));
      upperClamp.position.set(7.4 * s, -2.3 * s, 0);
      grp.add(upperClamp);

      const upperShim = new THREE.Mesh(upperShimGeo, lowerBlockMat);
      upperShim.position.set(7.4 * s, -6.25 * s, 0);
      grp.add(upperShim);

      for (const side of [-1, 1]) {
        const bolt = new THREE.Mesh(m3BoltGeo, stainlessBoltMat);
        bolt.position.set(7.4 * s, -7.8 * s, side * 4.5 * s);
        grp.add(bolt);
      }

      const alignPin = new THREE.Mesh(alignPinGeo, alignPinMat);
      alignPin.position.set(3.4 * s, -5.2 * s, 0);
      grp.add(alignPin);

      // Thin 301 stainless spring-steel leaf flexure blade with stepped flexure profile
      const blade = tag(new THREE.Mesh(leafBladeGeo, leafBladeMat));
      grp.add(blade);

      // Stepped dark-anodized CNC lower mount block bolted to the subframe
      const lowerStep = new THREE.Mesh(lowerStepGeo, upperClampMat);
      lowerStep.position.set(-8.6 * s, -7.0 * s, 0);
      grp.add(lowerStep);

      const lowerBase = tag(new THREE.Mesh(lowerBaseGeo, lowerBlockMat));
      lowerBase.position.set(-8.2 * s, -11.4 * s, 0);
      grp.add(lowerBase);

      for (const side of [-1, 1]) {
        const lowerBolt = new THREE.Mesh(m3BoltGeo, stainlessBoltMat);
        lowerBolt.position.set(-8.6 * s, -5.8 * s, side * 4.5 * s);
        grp.add(lowerBolt);
      }

      placeMountGroup(grp, sp);
    }
  } else if (bedMountType === "floating_4pt") {
    // 4-Pt Floating — Anchor #0 Fixed Shoulder Bolt + Anchors #1–3 Radially Slotted Tabs (12 mm)
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

    const fixedFlangeGeo = new THREE.CylinderGeometry(10.0 * s, 10.0 * s, 2.0 * s, 18);
    const fixedBushingGeo = new THREE.CylinderGeometry(8.0 * s, 8.6 * s, 8.4 * s, 18);
    const shoulderBoltGeo = new THREE.CylinderGeometry(3.8 * s, 3.8 * s, 12.6 * s, 14);
    const boltHeadGeo = new THREE.CylinderGeometry(5.2 * s, 5.2 * s, 2.8 * s, 14);

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

    const slottedTabGeo = hasFloatingCorners
      ? new THREE.BoxGeometry(24.0 * s, 3.6 * s, 12.4 * s)
      : null;
    const slotRecessGeo = hasFloatingCorners
      ? new THREE.BoxGeometry(12.2 * s, 3.9 * s, 3.8 * s)
      : null;
    const brassWasherGeo = hasFloatingCorners
      ? new THREE.CylinderGeometry(7.0 * s, 7.0 * s, 1.8 * s, 18)
      : null;
    const ptfeRingGeo = hasFloatingCorners
      ? new THREE.CylinderGeometry(7.6 * s, 7.6 * s, 1.1 * s, 18)
      : null;
    const lowerSpacerGeo = hasFloatingCorners
      ? new THREE.CylinderGeometry(6.0 * s, 6.4 * s, 4.6 * s, 16)
      : null;

    supportPtsThree.forEach((sp, idx) => {
      const grp = new THREE.Group();
      grp.rotation.y = radialYawToCenter(sp, bedBaseCenter);

      if (idx === 0) {
        // Anchor #0 (front-left): solid stainless fixed anchor shoulder bolt + machined steel locator bushing
        const flange = tag(new THREE.Mesh(fixedFlangeGeo, fixedBushingMat));
        flange.position.y = -1.0 * s;
        grp.add(flange);

        const bushing = tag(new THREE.Mesh(fixedBushingGeo, fixedBushingMat));
        bushing.position.y = -6.2 * s;
        grp.add(bushing);

        const botFlange = new THREE.Mesh(fixedFlangeGeo, fixedBushingMat);
        botFlange.position.y = -11.0 * s;
        grp.add(botFlange);

        const bolt = new THREE.Mesh(shoulderBoltGeo, stainlessMat);
        bolt.position.y = -6.3 * s;
        grp.add(bolt);

        const head = new THREE.Mesh(boltHeadGeo, stainlessMat);
        head.position.y = -13.4 * s;
        grp.add(head);
      } else if (
        slottedTabGeo &&
        slotRecessGeo &&
        brassWasherGeo &&
        ptfeRingGeo &&
        lowerSpacerGeo &&
        tabMat &&
        slotDarkMat &&
        brassWasherMat &&
        ptfeGlideMat
      ) {
        // Anchors #1–3: radially oriented slotted aluminum mount tabs + dark slot + brass/PTFE glide washer
        const tab = tag(new THREE.Mesh(slottedTabGeo, tabMat));
        tab.position.y = -2.2 * s;
        grp.add(tab);

        const slotRecess = new THREE.Mesh(slotRecessGeo, slotDarkMat);
        slotRecess.position.y = -2.2 * s;
        grp.add(slotRecess);

        const ptfeWasher = new THREE.Mesh(ptfeRingGeo, ptfeGlideMat);
        ptfeWasher.position.y = -4.55 * s;
        grp.add(ptfeWasher);

        const brassWasher = tag(new THREE.Mesh(brassWasherGeo, brassWasherMat));
        brassWasher.position.y = -6.0 * s;
        grp.add(brassWasher);

        const lowerSpacer = tag(new THREE.Mesh(lowerSpacerGeo, tabMat));
        lowerSpacer.position.y = -9.2 * s;
        grp.add(lowerSpacer);

        const bolt = new THREE.Mesh(shoulderBoltGeo, stainlessMat);
        bolt.position.y = -6.3 * s;
        grp.add(bolt);

        const head = new THREE.Mesh(boltHeadGeo, stainlessMat);
        head.position.y = -12.9 * s;
        grp.add(head);
      }

      placeMountGroup(grp, sp);
    });
  } else if (bedMountType === "silicone_4pt") {
    // 4-Pt 70A Matte Orange Silicone Isolator Columns (16 mm)
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

    const columnGeo = createSiliconeColumnGeometry(7.2 * s, 8.8 * s, -1.7 * s, -14.3 * s);
    const topWasherGeo = new THREE.CylinderGeometry(8.2 * s, 9.2 * s, 1.7 * s, 20);
    const botWasherGeo = new THREE.CylinderGeometry(9.2 * s, 8.2 * s, 1.7 * s, 20);
    const cskHeadGeo = new THREE.CylinderGeometry(3.8 * s, 2.1 * s, 2.2 * s, 14);
    const studGeo = new THREE.CylinderGeometry(2.0 * s, 2.0 * s, 3.2 * s, 10);

    for (const sp of supportPtsThree) {
      const grp = new THREE.Group();

      const topWasher = new THREE.Mesh(topWasherGeo, washerMat);
      topWasher.position.y = -0.85 * s;
      grp.add(topWasher);

      const column = tag(new THREE.Mesh(columnGeo, siliconeMat));
      grp.add(column);

      const botWasher = new THREE.Mesh(botWasherGeo, washerMat);
      botWasher.position.y = -15.15 * s;
      grp.add(botWasher);

      // Countersunk M4 screw head + lower threaded stud
      const cskHead = new THREE.Mesh(cskHeadGeo, screwMat);
      cskHead.position.y = -17.0 * s;
      grp.add(cskHead);

      const stud = new THREE.Mesh(studGeo, screwMat);
      stud.position.y = -19.2 * s;
      grp.add(stud);

      placeMountGroup(grp, sp);
    }
  } else if (bedMountType === "springs_4pt") {
    // 4-Pt Yellow ISO Die Springs (20 mm) — True Continuous Helical Coil + Knurled Leveling Thumbwheel
    const springMat = makeMat({
      color: 0xfacc15, // matte yellow ISO die-spring steel
      metalness: 0.34,
      roughness: 0.42,
      sensTint: true,
    });
    const cupMat = makeMat({
      color: 0x334155, // upper & lower dark steel spring cups
      metalness: 0.74,
      roughness: 0.38,
    });
    const rodMat = makeMat({
      color: 0xb8c4d4,
      metalness: 0.86,
      roughness: 0.24,
    });
    const wheelMat = makeMat({
      color: 0x1e293b, // matte black knurled POM/nylon leveling thumbwheel
      metalness: 0.14,
      roughness: 0.62,
    });
    const brassInsertMat = makeMat({
      color: 0xd4a338, // brass threaded insert hub in thumbwheel
      metalness: 0.84,
      roughness: 0.24,
    });

    const wireR = 1.35 * s;
    const helixGeo = createHelicalSpringGeometry(
      5.8 * s,
      wireR,
      -2.2 * s - wireR,
      -17.8 * s + wireR,
      5.5
    );
    const topCupGeo = new THREE.CylinderGeometry(8.4 * s, 8.9 * s, 2.2 * s, 18);
    const botCupGeo = new THREE.CylinderGeometry(8.9 * s, 8.4 * s, 2.2 * s, 18);
    const centerRodGeo = new THREE.CylinderGeometry(1.9 * s, 1.9 * s, 25.0 * s, 10);
    const thumbwheelGeo = createKnurledThumbwheelGeometry(13.8 * s, 1.2 * s, 18, 3.6 * s);
    const brassHubGeo = new THREE.CylinderGeometry(4.2 * s, 4.2 * s, 4.4 * s, 14);

    for (const sp of supportPtsThree) {
      const grp = new THREE.Group();

      const topCup = new THREE.Mesh(topCupGeo, cupMat);
      topCup.position.y = -1.1 * s;
      grp.add(topCup);

      const springCoil = tag(new THREE.Mesh(helixGeo, springMat));
      grp.add(springCoil);

      const botCup = new THREE.Mesh(botCupGeo, cupMat);
      botCup.position.y = -18.9 * s;
      grp.add(botCup);

      const guideRod = new THREE.Mesh(centerRodGeo, rodMat);
      guideRod.position.y = -12.5 * s;
      grp.add(guideRod);

      // Ribbed black knurled leveling thumbwheel + brass threaded insert underneath corner
      const wheel = tag(new THREE.Mesh(thumbwheelGeo, wheelMat));
      wheel.position.y = -22.2 * s;
      grp.add(wheel);

      const brassHub = new THREE.Mesh(brassHubGeo, brassInsertMat);
      brassHub.position.y = -22.2 * s;
      grp.add(brassHub);

      placeMountGroup(grp, sp);
    }
  } else {
    // rigid_4pt — 4-Pt Rigid Hard-Bolted Machined Hexagonal Standoffs (14 mm)
    const hexStandoffMat = makeMat({
      color: 0xdce4ee, // machined hexagonal stainless/billet aluminum standoff
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
    const socketMat = makeMat({
      color: 0x1e293b,
      metalness: 0.5,
      roughness: 0.55,
    });

    const hexBodyGeo = new THREE.CylinderGeometry(7.2 * s, 7.2 * s, 10.4 * s, 6);
    const topFlangeGeo = new THREE.CylinderGeometry(9.4 * s, 10.2 * s, 1.8 * s, 20);
    const botFlangeGeo = new THREE.CylinderGeometry(10.2 * s, 9.4 * s, 1.8 * s, 20);
    const capScrewGeo = new THREE.CylinderGeometry(3.8 * s, 3.8 * s, 3.2 * s, 14);
    const hexRecessGeo = new THREE.CylinderGeometry(1.9 * s, 1.9 * s, 0.4 * s, 6);

    for (const sp of supportPtsThree) {
      const grp = new THREE.Group();

      const topWasher = new THREE.Mesh(topFlangeGeo, washerMat);
      topWasher.position.y = -0.9 * s;
      grp.add(topWasher);

      const hexStandoff = tag(new THREE.Mesh(hexBodyGeo, hexStandoffMat));
      hexStandoff.position.y = -7.0 * s;
      grp.add(hexStandoff);

      const botWasher = new THREE.Mesh(botFlangeGeo, washerMat);
      botWasher.position.y = -13.1 * s;
      grp.add(botWasher);

      const capScrew = new THREE.Mesh(capScrewGeo, screwMat);
      capScrew.position.y = -15.6 * s;
      grp.add(capScrew);

      const hexSocket = new THREE.Mesh(hexRecessGeo, socketMat);
      hexSocket.position.y = -17.15 * s;
      grp.add(hexSocket);

      placeMountGroup(grp, sp);
    }
  }

  return root;
}
