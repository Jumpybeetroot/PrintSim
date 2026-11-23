import * as THREE from "three";

const STRESS_STOP_0 = new THREE.Color("#2563eb");
const STRESS_STOP_1 = new THREE.Color("#06b6d4");
const STRESS_STOP_2 = new THREE.Color("#22c55e");
const STRESS_STOP_3 = new THREE.Color("#eab308");
const STRESS_STOP_4 = new THREE.Color("#ef4444");

/**
 * Scientific FEA stress / deflection colormap:
 * blue -> cyan -> green -> yellow -> red for normalized t in [0, 1].
 */
export function stressColorInto(t: number, out: THREE.Color): THREE.Color {
  const c = Math.max(0, Math.min(1, t));
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

export function stressColor(t: number): THREE.Color {
  return stressColorInto(t, new THREE.Color());
}

const SENS_NEG_MAX = new THREE.Color("#06b6d4"); // electric cyan (-1.0)
const SENS_NEG_MID = new THREE.Color("#38bdf8"); // sky (-0.5)
const SENS_NEUTRAL = new THREE.Color("#334155"); // slate (0.0)
const SENS_POS_MID = new THREE.Color("#f59e0b"); // amber (+0.5)
const SENS_POS_MAX = new THREE.Color("#ef4444"); // blazing crimson (+1.0)

/**
 * Diverging adjoint sensitivity colormap for score in [-1, +1]:
 * negative (lighten mass) = cyan/sky, neutral = slate, positive (stiffen) = amber/crimson.
 */
export function sensitivityColorInto(score: number, out: THREE.Color): THREE.Color {
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

export function sensitivityColor(score: number): THREE.Color {
  return sensitivityColorInto(score, new THREE.Color());
}

const BED_WARP_VALLEY = new THREE.Color("#3b82f6");
const BED_WARP_FLAT = new THREE.Color("#10b981");
const BED_WARP_CROWN = new THREE.Color("#f59e0b");
const BED_WARP_SEVERE = new THREE.Color("#ef4444");

/**
 * Thermal bed-plate bow/warp colormap for normalized height t in [0, 1]:
 * blue valley -> emerald flat -> amber crown -> red severe bow.
 */
export function bedWarpColorInto(t: number, out: THREE.Color): THREE.Color {
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

export function bedWarpColor(t: number): THREE.Color {
  return bedWarpColorInto(t, new THREE.Color());
}
