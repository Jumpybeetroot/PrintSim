import * as THREE from "three";

export interface ModalShellMesh {
  geometry: THREE.BufferGeometry;
  wireGeometry: THREE.BufferGeometry;
  vertNodeIds: string[];
}

export interface ElasticDeformMesh {
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

export const HOTEND_LUT_N = 64;
const HOTEND_PROFILE_CACHE: Record<string, { slope: Float32Array; disp: Float32Array }> = {};

export function getHotendProfileLUT(hotendMountStyle: string): { slope: Float32Array; disp: Float32Array } {
  const key =
    hotendMountStyle === "braced"
      ? "braced"
      : hotendMountStyle === "chube_6bolt_m3_18pcd"
        ? "chube"
        : "necked";
  const cached = HOTEND_PROFILE_CACHE[key];
  if (cached) return cached;

  const slope = new Float32Array(HOTEND_LUT_N + 1);
  const disp = new Float32Array(HOTEND_LUT_N + 1);

  if (key === "braced") {
    // Doubly supported hotend (top Port 1 crown + lower Ti Hotend_Brace through s ≈ 0.56):
    // Keep the braced heatsink span virtually rigid so Tricorn never clips Hotend_Brace,
    // with gentle elastic cantilever flexure restricted to the unbraced lower heater/nozzle tip.
    const kappa = new Float32Array(HOTEND_LUT_N + 1);
    for (let k = 0; k <= HOTEND_LUT_N; k++) {
      const s = k / HOTEND_LUT_N;
      if (s <= 0.55) {
        kappa[k] = 0.015;
      } else {
        const u = (s - 0.55) / 0.45;
        kappa[k] = 0.015 + u * u * (1.2 - 0.35 * u);
      }
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
  } else if (key === "chube") {
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

export function sampleProfileLUT(lut: Float32Array, s: number): number {
  const clamped = s <= 0 ? 0 : s >= 1 ? 1 : s;
  const f = clamped * HOTEND_LUT_N;
  const i0 = Math.floor(f);
  const i1 = i0 < HOTEND_LUT_N ? i0 + 1 : HOTEND_LUT_N;
  const frac = f - i0;
  return lut[i0] + (lut[i1] - lut[i0]) * frac;
}

export function extractRotVecFromQuat(
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

export function deformBackplateMesh(
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

export function deformBodyMesh(
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

export function deformHotendMesh(
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
