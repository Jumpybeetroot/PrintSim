import * as THREE from "three";
import { GeometryMember } from "../types";
import { stressColorInto } from "./colormaps";

export interface ContinuousBeamChain {
  nodeIds: string[];
  members: GeometryMember[];
  template: GeometryMember;
  isRod: boolean;
}

export interface ModalContinuousBeam {
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

/**
 * Merge collinear, matching-spec rectangular extrusions (e.g. col_*_low + col_*_top,
 * y_rail_*_f + y_rail_*_r, x_gantry_l + x_gantry_r, Cartesian Z-columns/base rails,
 * Delta towers) into single continuous multi-span chains.
 */
export function groupContinuousBeams(
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
export function updateContinuousBeamGeometry(
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

export function createContinuousBeamGeometry(
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

export function applyRodGradientColors(geo: THREE.BufferGeometry, val0: number, val1: number): void {
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
