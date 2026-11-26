import React, { useEffect, useMemo, useRef, useState } from "react";
import { PrinterSpecData, TimeDomainMetrologyData } from "../types";

export interface SurfaceTexturePreviewProps {
  metrology: TimeDomainMetrologyData;
  spec?: PrinterSpecData;
}

type FilamentFinishId = "silk_silver" | "matte_charcoal" | "silk_gold" | "matte_orange";

interface FilamentFinishSpec {
  id: FilamentFinishId;
  label: string;
  baseRgb: [number, number, number];
  specRgb: [number, number, number];
  ambient: number;
  diffuse: number;
  specular: number;
  shininess: number;
  layerContrast: number;
}

const FILAMENT_FINISHES: Record<FilamentFinishId, FilamentFinishSpec> = {
  silk_silver: {
    id: "silk_silver",
    label: "Silk Silver",
    baseRgb: [176, 189, 206],
    specRgb: [248, 252, 255],
    ambient: 0.22,
    diffuse: 0.62,
    specular: 0.88,
    shininess: 40,
    layerContrast: 0.34,
  },
  matte_charcoal: {
    id: "matte_charcoal",
    label: "Matte Charcoal",
    baseRgb: [62, 71, 86],
    specRgb: [158, 174, 198],
    ambient: 0.26,
    diffuse: 0.72,
    specular: 0.22,
    shininess: 10,
    layerContrast: 0.24,
  },
  silk_gold: {
    id: "silk_gold",
    label: "Silk Gold",
    baseRgb: [208, 164, 68],
    specRgb: [255, 244, 204],
    ambient: 0.22,
    diffuse: 0.62,
    specular: 0.9,
    shininess: 36,
    layerContrast: 0.34,
  },
  matte_orange: {
    id: "matte_orange",
    label: "Matte Orange",
    baseRgb: [216, 106, 38],
    specRgb: [255, 206, 170],
    ambient: 0.24,
    diffuse: 0.74,
    specular: 0.25,
    shininess: 12,
    layerContrast: 0.26,
  },
};

const WALL_LENGTH_MM = 25.0;
const WALL_HEIGHT_MM = 9.6;
const LAYER_HEIGHT_MM = 0.2;
const CANVAS_W = 320;
const CANVAS_H = 128;

function resolveBeltPitchMm(spec?: PrinterSpecData): number {
  const explicit = spec?.gantry.belt.pitch_mm;
  if (explicit && Number.isFinite(explicit) && explicit > 0) return explicit;
  const bType = (spec?.gantry.belt.type ?? "gt2_6mm").toLowerCase();
  if (bType.startsWith("at5")) return 5.0;
  if (bType.startsWith("3gt") || bType.startsWith("at3") || bType.startsWith("htd3m")) {
    return 3.0;
  }
  return 2.0;
}

interface WallProfileData {
  distMm: number[];
  unshapedUm: number[];
  shapedUm: number[];
}

function extractWallProfile(metrology: TimeDomainMetrologyData): WallProfileData | null {
  const trace = metrology.corner_trace;
  if (!trace) return null;

  const wd = trace.wall_dist_mm;
  const wu = trace.wall_dev_unshaped_um;
  const ws = trace.wall_dev_shaped_um;
  if (wd && wu && ws) {
    const n = Math.min(wd.length, wu.length, ws.length);
    if (n >= 2) {
      return {
        distMm: wd.slice(0, n),
        unshapedUm: wu.slice(0, n),
        shapedUm: ws.slice(0, n),
      };
    }
  }

  // Fallback: extract post-corner leg normal deviation from (ideal, actual, unshaped) XY trace
  const ix = trace.ideal_x_mm || [];
  const iy = trace.ideal_y_mm || [];
  const ax = trace.actual_x_mm || [];
  const ay = trace.actual_y_mm || [];
  const ux = trace.unshaped_x_mm || [];
  const uy = trace.unshaped_y_mm || [];
  const n = Math.min(ix.length, iy.length, ax.length, ay.length, ux.length, uy.length);
  if (n < 4) return null;

  let cornerIdx = Math.floor(n * 0.35);
  let bestTurn = -1;
  for (let i = 1; i < n - 1; i++) {
    const dx1 = ix[i] - ix[i - 1];
    const dy1 = iy[i] - iy[i - 1];
    const dx2 = ix[i + 1] - ix[i];
    const dy2 = iy[i + 1] - iy[i];
    const l1 = Math.hypot(dx1, dy1);
    const l2 = Math.hypot(dx2, dy2);
    if (l1 < 1e-6 || l2 < 1e-6) continue;
    const dot = Math.max(-1, Math.min(1, (dx1 * dx2 + dy1 * dy2) / (l1 * l2)));
    const turn = Math.acos(dot);
    if (turn > bestTurn) {
      bestTurn = turn;
      cornerIdx = i;
    }
  }

  const cx = ix[cornerIdx];
  const cy = iy[cornerIdx];
  const endX = ix[n - 1] - cx;
  const endY = iy[n - 1] - cy;
  const legLen = Math.hypot(endX, endY);
  if (legLen < 1e-4) return null;

  const tx = endX / legLen;
  const ty = endY / legLen;
  const nx = -ty;
  const ny = tx;

  const distMm: number[] = [];
  const unshapedUm: number[] = [];
  const shapedUm: number[] = [];

  for (let i = cornerIdx; i < n; i++) {
    const dIdeal = (ix[i] - cx) * tx + (iy[i] - cy) * ty;
    if (dIdeal < 0) continue;
    const devShMm = (ax[i] - cx) * nx + (ay[i] - cy) * ny;
    const devUnMm = (ux[i] - cx) * nx + (uy[i] - cy) * ny;
    distMm.push(dIdeal);
    shapedUm.push(devShMm * 1000);
    unshapedUm.push(devUnMm * 1000);
  }

  if (distMm.length < 2) return null;
  return { distMm, unshapedUm, shapedUm };
}

function sampleLinear(xs: number[], ys: number[], x: number): number {
  const n = xs.length;
  if (n === 0) return 0;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const span = xs[hi] - xs[lo];
  const t = span > 1e-9 ? (x - xs[lo]) / span : 0;
  return ys[lo] + (ys[hi] - ys[lo]) * t;
}

export const SurfaceTexturePreview: React.FC<SurfaceTexturePreviewProps> = ({
  metrology,
  spec,
}) => {
  const [finishId, setFinishId] = useState<FilamentFinishId>("silk_silver");
  const [lightAngleDeg, setLightAngleDeg] = useState<number>(75);
  const [rippleGain, setRippleGain] = useState<number>(15);
  const [showVfa, setShowVfa] = useState<boolean>(false);
  const [splitPct, setSplitPct] = useState<number>(50);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef<boolean>(false);

  const wallProfile = useMemo(() => extractWallProfile(metrology), [metrology]);

  const vfaGeometry = useMemo(() => {
    const beltPitchMm = resolveBeltPitchMm(spec);
    const pulleyTeeth = Math.max(10, spec?.gantry.belt.pulley_teeth ?? 20);
    const stepsPerRev = Math.max(100, spec?.kinematics.xy_stepper.steps_per_rev ?? 200);
    const circMm = pulleyTeeth * beltPitchMm;
    const stepWaveMm = circMm / stepsPerRev;
    const elecWaveMm = stepWaveMm * 4.0;
    return {
      beltWaveMm: beltPitchMm,
      elecWaveMm,
      stepWaveMm,
      is09Deg: stepsPerRev >= 350,
    };
  }, [spec]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !wallProfile) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const pxW = Math.round(CANVAS_W * dpr);
    const pxH = Math.round(CANVAS_H * dpr);
    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW;
      canvas.height = pxH;
    }

    const finish = FILAMENT_FINISHES[finishId];
    const thetaRad = (lightAngleDeg * Math.PI) / 180;
    const lx0 = Math.sin(thetaRad);
    const ly0 = -0.22;
    const lz0 = Math.cos(thetaRad);
    const lNorm = Math.hypot(lx0, ly0, lz0) || 1;
    const lx = lx0 / lNorm;
    const ly = ly0 / lNorm;
    const lz = lz0 / lNorm;

    // Blinn-Phong half vector with V = (0, 0, 1)
    const hx0 = lx;
    const hy0 = ly;
    const hz0 = lz + 1.0;
    const hNorm = Math.hypot(hx0, hy0, hz0) || 1;
    const hx = hx0 / hNorm;
    const hy = hy0 / hNorm;
    const hz = hz0 / hNorm;

    const splitX = Math.round((splitPct / 100) * pxW);
    const wallTop = Math.round(18 * dpr);
    const wallBottom = pxH - Math.round(18 * dpr);
    const wallH = Math.max(1, wallBottom - wallTop);

    // Precompute height profile and horizontal slope dh/dx (mm/mm) for unshaped & shaped
    const dxMm = WALL_LENGTH_MM / Math.max(pxW - 1, 1);
    const unshapedH = new Float32Array(pxW);
    const shapedH = new Float32Array(pxW);

    const twoPi = Math.PI * 2;
    const beltAmpUm = vfaGeometry.beltWaveMm >= 3 ? 4.2 : 3.2;
    const elecAmpUm = vfaGeometry.is09Deg ? 2.0 : 4.6;
    const stepAmpUm = vfaGeometry.is09Deg ? 0.7 : 1.4;

    for (let x = 0; x < pxW; x++) {
      const distMm = (x / Math.max(pxW - 1, 1)) * WALL_LENGTH_MM;
      let vfaUm = 0;
      if (showVfa) {
        vfaUm =
          beltAmpUm * Math.sin((twoPi * distMm) / vfaGeometry.beltWaveMm) +
          elecAmpUm * Math.sin((twoPi * distMm) / vfaGeometry.elecWaveMm + 0.6) +
          stepAmpUm * Math.sin((twoPi * distMm) / vfaGeometry.stepWaveMm + 1.2);
      }
      unshapedH[x] =
        (sampleLinear(wallProfile.distMm, wallProfile.unshapedUm, distMm) + vfaUm) *
        rippleGain *
        0.001;
      shapedH[x] =
        (sampleLinear(wallProfile.distMm, wallProfile.shapedUm, distMm) + vfaUm) *
        rippleGain *
        0.001;
    }

    const slopeUn = new Float32Array(pxW);
    const slopeSh = new Float32Array(pxW);
    for (let x = 0; x < pxW; x++) {
      const xm = Math.max(0, x - 1);
      const xp = Math.min(pxW - 1, x + 1);
      const spanMm = Math.max(1, xp - xm) * dxMm;
      slopeUn[x] = (unshapedH[xp] - unshapedH[xm]) / spanMm;
      slopeSh[x] = (shapedH[xp] - shapedH[xm]) / spanMm;
    }

    // Precompute vertical layer-line cylindrical normal component & groove ambient occlusion
    const layerNy = new Float32Array(pxH);
    const layerAo = new Float32Array(pxH);
    for (let y = 0; y < pxH; y++) {
      if (y < wallTop || y >= wallBottom) continue;
      const zMm = ((y - wallTop) / wallH) * WALL_HEIGHT_MM;
      const layerFrac = (zMm / LAYER_HEIGHT_MM) % 1;
      // Cylindrical bead profile across each 0.2 mm extruded filament layer
      const beadPos = layerFrac * 2 - 1; // [-1, 1]
      layerNy[y] = beadPos * finish.layerContrast * 1.45;
      // Subtle ambient occlusion inside inter-layer valley near |beadPos| -> 1
      const groove = beadPos * beadPos;
      layerAo[y] = 1.0 - finish.layerContrast * 0.42 * groove * groove;
    }

    const img = ctx.createImageData(pxW, pxH);
    const data = img.data;
    const [br, bg, bb] = finish.baseRgb;
    const [sr, sg, sb] = finish.specRgb;

    for (let y = 0; y < pxH; y++) {
      const inWall = y >= wallTop && y < wallBottom;
      const ny = inWall ? layerNy[y] : 0;
      const ao = inWall ? layerAo[y] : 1;
      const rowOffset = y * pxW * 4;

      if (!inWall) {
        for (let x = 0; x < pxW; x++) {
          const idx = rowOffset + x * 4;
          data[idx] = 11;
          data[idx + 1] = 17;
          data[idx + 2] = 27;
          data[idx + 3] = 255;
        }
        continue;
      }

      for (let x = 0; x < pxW; x++) {
        const isShaped = x >= splitX;
        const slope = isShaped ? slopeSh[x] : slopeUn[x];
        // Slight corner-edge bevel highlight at x = 0..2mm
        const edgeBevel = x < 4 * dpr ? -0.35 * (1 - x / (4 * dpr)) : 0;
        const nx = -(slope * 2.2) + edgeBevel;
        const nz = 1.0;
        const invLen = 1.0 / Math.hypot(nx, ny, nz);
        const nnx = nx * invLen;
        const nny = ny * invLen;
        const nnz = nz * invLen;

        const ndotl = Math.max(0, nnx * lx + nny * ly + nnz * lz);
        const ndoth = Math.max(0, nnx * hx + nny * hy + nnz * hz);
        const specTerm = Math.pow(ndoth, finish.shininess) * finish.specular;
        const lit = (finish.ambient + finish.diffuse * ndotl) * ao;

        const idx = rowOffset + x * 4;
        data[idx] = Math.min(255, Math.max(0, Math.round(br * lit + sr * specTerm * ao)));
        data[idx + 1] = Math.min(255, Math.max(0, Math.round(bg * lit + sg * specTerm * ao)));
        data[idx + 2] = Math.min(255, Math.max(0, Math.round(bb * lit + sb * specTerm * ao)));
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);

    // Draw crisp vector overlays (ruler ticks, split curtain, labels)
    ctx.save();
    ctx.scale(dpr, dpr);

    // Top/bottom wall seam borders
    ctx.strokeStyle = "rgba(37, 47, 66, 0.9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 18);
    ctx.lineTo(CANVAS_W, 18);
    ctx.moveTo(0, CANVAS_H - 18);
    ctx.lineTo(CANVAS_W, CANVAS_H - 18);
    ctx.stroke();

    // Bottom ruler ticks (0..25 mm)
    ctx.fillStyle = "#8c9bb3";
    ctx.strokeStyle = "#37445c";
    ctx.font = "500 9px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let mm = 0; mm <= 25; mm += 5) {
      const x = (mm / WALL_LENGTH_MM) * CANVAS_W;
      ctx.beginPath();
      ctx.moveTo(x, CANVAS_H - 18);
      ctx.lineTo(x, CANVAS_H - 13);
      ctx.stroke();
      const labelX = mm === 0 ? 11 : mm === 25 ? CANVAS_W - 15 : x;
      ctx.fillText(mm === 0 ? "0 mm" : mm === 25 ? "25 mm" : `${mm}`, labelX, CANVAS_H - 7);
    }

    // Top-left Unshaped and Top-right Shaped labels
    const splitCssX = (splitPct / 100) * CANVAS_W;
    ctx.font = "600 9.5px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    if (splitCssX > 66) {
      ctx.fillStyle = "#fca5a5";
      ctx.textAlign = "left";
      ctx.fillText("UNSHAPED", 6, 9.5);
    }
    if (CANVAS_W - splitCssX > 56) {
      ctx.fillStyle = "#7ee2b0";
      ctx.textAlign = "right";
      ctx.fillText("SHAPED", CANVAS_W - 6, 9.5);
    }

    // Vertical split curtain line + grab handle
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(splitCssX, 0);
    ctx.lineTo(splitCssX, CANVAS_H - 18);
    ctx.stroke();

    const handleY = 18 + (CANVAS_H - 36) * 0.5;
    ctx.fillStyle = "#0d121b";
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.roundRect(splitCssX - 9, handleY - 11, 18, 22, 9);
    ctx.fill();
    ctx.stroke();

    // Left/right arrows inside handle
    ctx.fillStyle = "#bfe3fb";
    ctx.font = "700 9px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("↔", splitCssX, handleY + 0.5);

    ctx.restore();
  }, [wallProfile, finishId, lightAngleDeg, rippleGain, showVfa, splitPct, vfaGeometry]);

  if (!wallProfile) return null;

  const updateSplitFromClientX = (clientX: number) => {
    const el = stageRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSplitPct(Math.max(5, Math.min(95, pct)));
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    updateSplitFromClientX(e.clientX);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    updateSplitFromClientX(e.clientX);
  };

  const handlePointerEnd = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      setSplitPct((v) => Math.max(5, v - 5));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      setSplitPct((v) => Math.min(95, v + 5));
    } else if (e.key === "Home") {
      e.preventDefault();
      setSplitPct(5);
    } else if (e.key === "End") {
      e.preventDefault();
      setSplitPct(95);
    }
  };

  return (
    <div className="metrology-subblock">
      <div className="metrology-subhead">
        <span>Printed Wall Surface Preview</span>
        <span className="metrology-submeta">Drag curtain · 25 mm wall</span>
      </div>

      <div
        ref={stageRef}
        className="wall-preview-stage"
        role="slider"
        tabIndex={0}
        aria-label="Unshaped vs. Shaped printed wall comparison curtain (%)"
        aria-valuemin={5}
        aria-valuemax={95}
        aria-valuenow={Math.round(splitPct)}
        title="Drag left or right (or use Arrow keys) to compare unshaped vs. input-shaped ringing on a lit 25 mm FDM wall"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerEnd}
        onKeyDown={handleKeyDown}
      >
        <canvas ref={canvasRef} className="wall-preview-canvas" />
      </div>

      <div className="wall-preview-controls">
        <div className="field-grid-2">
          <div className="field-row">
            <label htmlFor="wall-finish-select">Filament Finish</label>
            <select
              id="wall-finish-select"
              value={finishId}
              onChange={(e) => setFinishId(e.target.value as FilamentFinishId)}
            >
              {Object.values(FILAMENT_FINISHES).map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field-row">
            <div className="wall-slider-label-row">
              <label htmlFor="wall-light-angle">Grazing Light (°)</label>
              <span className="wall-slider-val">{lightAngleDeg}°</span>
            </div>
            <input
              id="wall-light-angle"
              type="range"
              min={20}
              max={85}
              step={1}
              value={lightAngleDeg}
              onChange={(e) => setLightAngleDeg(Number(e.target.value))}
              title="Grazing light angle from wall normal (higher angles exaggerate surface ripples)"
            />
          </div>
        </div>

        <div className="field-row">
          <div className="wall-slider-label-row">
            <label htmlFor="wall-ripple-gain">Ripple Gain (×)</label>
            <span className="wall-slider-val">{rippleGain}×</span>
          </div>
          <input
            id="wall-ripple-gain"
            type="range"
            min={1}
            max={30}
            step={1}
            value={rippleGain}
            onChange={(e) => setRippleGain(Number(e.target.value))}
            title="Visual displacement magnification for ringing and VFA surface ripples"
          />
        </div>

        <div className="wall-vfa-row">
          <label className="checkbox-row" htmlFor="wall-vfa-toggle">
            <input
              id="wall-vfa-toggle"
              type="checkbox"
              checked={showVfa}
              onChange={(e) => setShowVfa(e.target.checked)}
            />
            <span>Show Belt/Motor VFA Ripples</span>
          </label>
          {showVfa && (
            <span
              className="wall-vfa-meta"
              title="Belt tooth pitch wavelength and motor electrical-pole VFA wavelength (unaffected by input shaping)"
            >
              λ {vfaGeometry.beltWaveMm.toFixed(1)} mm belt · {vfaGeometry.elecWaveMm.toFixed(2)} mm elec
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
