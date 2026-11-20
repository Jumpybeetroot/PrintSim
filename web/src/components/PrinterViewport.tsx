import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SimulationResponse, SolidFEAResponse, ViewMode } from "../types";

interface Props {
  sim: SimulationResponse | null;
  solidFea: SolidFEAResponse | null;
  viewMode: ViewMode;
  onViewModeChange: (m: ViewMode) => void;
  selectedModeIdx: number;
  onSelectModeIdx: (idx: number) => void;
  feaScale: number;
  onFeaScaleChange: (s: number) => void;
  layerLimit: number;
  onLayerLimitChange: (l: number) => void;
}

const SCALE = 0.01; // 100 mm = 1 Three.js unit

function stressColor(t: number): THREE.Color {
  const c = Math.max(0, Math.min(1, t));
  const color = new THREE.Color();
  // Scientific blue -> cyan -> green -> yellow -> red
  if (c < 0.25) {
    color.lerpColors(new THREE.Color("#2563eb"), new THREE.Color("#06b6d4"), c / 0.25);
  } else if (c < 0.5) {
    color.lerpColors(new THREE.Color("#06b6d4"), new THREE.Color("#22c55e"), (c - 0.25) / 0.25);
  } else if (c < 0.75) {
    color.lerpColors(new THREE.Color("#22c55e"), new THREE.Color("#eab308"), (c - 0.5) / 0.25);
  } else {
    color.lerpColors(new THREE.Color("#eab308"), new THREE.Color("#ef4444"), (c - 0.75) / 0.25);
  }
  return color;
}

function toThree(ptMm: [number, number, number], cx: number, cy: number): THREE.Vector3 {
  return new THREE.Vector3((ptMm[0] - cx) * SCALE, ptMm[2] * SCALE, -(ptMm[1] - cy) * SCALE);
}

function orientBeamMesh(mesh: THREE.Mesh, p0: THREE.Vector3, p1: THREE.Vector3): void {
  const dir = new THREE.Vector3().subVectors(p1, p0);
  const len = Math.max(dir.length(), 1e-4);
  mesh.position.copy(p0).add(p1).multiplyScalar(0.5);
  mesh.scale.set(1, len, 1);
  const up = new THREE.Vector3(0, 1, 0);
  const quat = new THREE.Quaternion().setFromUnitVectors(up, dir.normalize());
  mesh.quaternion.copy(quat);
}

export const PrinterViewport: React.FC<Props> = ({
  sim,
  solidFea,
  viewMode,
  onViewModeChange,
  selectedModeIdx,
  onSelectModeIdx,
  feaScale,
  onFeaScaleChange,
  layerLimit,
  onLayerLimitChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer?: THREE.WebGLRenderer;
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    controls?: OrbitControls;
    dynamicGroup?: THREE.Group;
    animFrame?: number;
    modalMeshes: Array<{
      mesh: THREE.Mesh;
      startId: string;
      endId: string;
    }>;
    modalToolhead?: THREE.Group;
    basePositions: Record<string, THREE.Vector3>;
    modalVectors: Record<string, THREE.Vector3>;
    isModal: boolean;
  }>({
    modalMeshes: [],
    basePositions: {},
    modalVectors: {},
    isModal: false,
  });

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

    const onResize = () => {
      if (!el) return;
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    };
    window.addEventListener("resize", onResize);

    const clock = new THREE.Clock();
    const animate = () => {
      stateRef.current.animFrame = requestAnimationFrame(animate);
      controls.update();

      if (stateRef.current.isModal && stateRef.current.modalMeshes.length > 0) {
        const t = clock.getElapsedTime();
        const phase = Math.sin(t * 7.5) * 0.38;
        const { basePositions, modalVectors, modalMeshes, modalToolhead } = stateRef.current;

        const getPos = (id: string) => {
          const b = basePositions[id];
          const v = modalVectors[id];
          if (!b) return new THREE.Vector3();
          if (!v) return b.clone();
          return b.clone().addScaledVector(v, phase);
        };

        for (const item of modalMeshes) {
          orientBeamMesh(item.mesh, getPos(item.startId), getPos(item.endId));
        }
        if (modalToolhead) {
          const thPos = getPos("toolhead");
          modalToolhead.position.copy(thPos);
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener("resize", onResize);
      if (stateRef.current.animFrame) cancelAnimationFrame(stateRef.current.animFrame);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === el) el.removeChild(renderer.domElement);
    };
  }, []);

  // Rebuild 3D geometry when simulation results or visualization mode changes
  useEffect(() => {
    const group = stateRef.current.dynamicGroup;
    if (!group) return;

    // Clear previous meshes
    while (group.children.length > 0) {
      const obj = group.children[0];
      group.remove(obj);
      obj.traverse((child) => {
        if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose();
        const mat = (child as THREE.Mesh).material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
    }

    stateRef.current.modalMeshes = [];
    stateRef.current.modalToolhead = undefined;
    stateRef.current.isModal = viewMode === "modal";

    // Render 3D Solid Hex8 Voxel FEA mode
    if (viewMode === "solid_part") {
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
      return;
    }

    if (!sim) return;

    const geom = sim.geometry;
    const cx = geom.frame_dims_mm[0] * 0.5;
    const cy = geom.frame_dims_mm[1] * 0.5;

    const basePos: Record<string, THREE.Vector3> = {};
    const deformedPos: Record<string, THREE.Vector3> = {};

    for (const n of geom.nodes) {
      const p0 = toThree(n.position_mm, cx, cy);
      basePos[n.id] = p0;

      const disp = sim.static_fea.nodal_displacements_mm[n.id] || [0, 0, 0];
      const dispVec = new THREE.Vector3(disp[0] * SCALE, disp[2] * SCALE, -disp[1] * SCALE);
      deformedPos[n.id] = p0.clone().addScaledVector(dispVec, feaScale);
    }
    stateRef.current.basePositions = basePos;

    // Modal displacement vectors
    const activeMode = sim.modal_fea.modes[selectedModeIdx] || sim.modal_fea.modes[0];
    const mVecs: Record<string, THREE.Vector3> = {};
    if (activeMode) {
      for (const [nid, vec] of Object.entries(activeMode.nodal_vectors)) {
        mVecs[nid] = new THREE.Vector3(vec[0], vec[2], -vec[1]);
      }
    }
    stateRef.current.modalVectors = mVecs;

    // Stress lookup
    const stressMap: Record<string, number> = {};
    const maxStress = Math.max(sim.static_fea.max_von_mises_mpa, 0.05);
    for (const ms of sim.static_fea.member_stresses) {
      stressMap[ms.member_id] = ms.von_mises_mpa / maxStress;
    }

    // Build structural members
    const frameOpacity = viewMode === "toolpath" ? 0.25 : 1.0;

    for (const m of geom.members) {
      const w = Math.max(m.width_mm * SCALE, 0.06);
      const h = Math.max(m.height_mm * SCALE, 0.06);
      const geo =
        m.shape === "rod"
          ? new THREE.CylinderGeometry(w * 0.5, w * 0.5, 1, 10)
          : new THREE.BoxGeometry(w, 1, h);

      let color = new THREE.Color(m.color_hex);
      if (viewMode === "fea_stress") {
        color = stressColor(stressMap[m.id] ?? 0);
      } else if (viewMode === "modal") {
        const amp0 = mVecs[m.start_id]?.length() ?? 0;
        const amp1 = mVecs[m.end_id]?.length() ?? 0;
        color = stressColor(0.5 * (amp0 + amp1));
      } else if (m.role === "brace") {
        color = new THREE.Color("#38bdf8");
      } else if (m.role === "gantry") {
        color = m.material.includes("carbon") ? new THREE.Color("#1f2937") : new THREE.Color("#f59e0b");
      }

      const mat = new THREE.MeshStandardMaterial({
        color,
        metalness: m.material.includes("carbon") ? 0.15 : 0.55,
        roughness: m.material.includes("carbon") ? 0.65 : 0.35,
        transparent: frameOpacity < 1,
        opacity: frameOpacity,
      });

      const mesh = new THREE.Mesh(geo, mat);
      const p0 = viewMode === "fea_stress" ? deformedPos[m.start_id] : basePos[m.start_id];
      const p1 = viewMode === "fea_stress" ? deformedPos[m.end_id] : basePos[m.end_id];
      orientBeamMesh(mesh, p0, p1);
      group.add(mesh);

      if (viewMode === "modal") {
        stateRef.current.modalMeshes.push({ mesh, startId: m.start_id, endId: m.end_id });
      }

      // Undeformed ghost reference in FEA / Modal mode
      if (viewMode === "fea_stress" || viewMode === "modal") {
        const lineGeo = new THREE.BufferGeometry().setFromPoints([basePos[m.start_id], basePos[m.end_id]]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x475569, transparent: true, opacity: 0.35 });
        group.add(new THREE.Line(lineGeo, lineMat));
      }
    }

    // Foot anchor markers
    for (const n of geom.nodes) {
      if (n.fixed) {
        const pad = new THREE.Mesh(
          new THREE.CylinderGeometry(0.14, 0.17, 0.08, 12),
          new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.5 })
        );
        pad.position.copy(basePos[n.id]);
        group.add(pad);
      }
    }

    // Heated bed plate
    const [bedX, bedY] = geom.bed_size_mm;
    const bedGeo = new THREE.BoxGeometry(bedX * SCALE, 0.08, bedY * SCALE);
    const bedMat = new THREE.MeshStandardMaterial({
      color: 0x2a3444,
      metalness: 0.4,
      roughness: 0.45,
      transparent: viewMode === "fea_stress" || viewMode === "modal",
      opacity: viewMode === "fea_stress" || viewMode === "modal" ? 0.35 : 0.95,
    });
    const bedMesh = new THREE.Mesh(bedGeo, bedMat);
    bedMesh.position.copy(toThree(geom.bed_center_mm, cx, cy));
    bedMesh.position.y -= 0.04;
    group.add(bedMesh);

    // Build volume wireframe
    const [bx, by, bz] = geom.build_volume_mm;
    const bvBox = new THREE.BoxGeometry(bx * SCALE, bz * SCALE, by * SCALE);
    const bvEdges = new THREE.EdgesGeometry(bvBox);
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

    // Toolhead assembly
    const thGroup = new THREE.Group();
    const [ex, ey, ez] = geom.toolhead_envelope_mm;
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(ex * SCALE, ez * SCALE * 0.72, ey * SCALE),
      new THREE.MeshStandardMaterial({
        color: 0xe63946,
        roughness: 0.4,
        transparent: viewMode === "toolpath",
        opacity: viewMode === "toolpath" ? 0.4 : 0.95,
      })
    );
    body.position.y = -ez * SCALE * 0.2;
    thGroup.add(body);

    const nozzle = new THREE.Mesh(
      new THREE.ConeGeometry(0.08, 0.22, 10),
      new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.85, roughness: 0.2 })
    );
    nozzle.rotation.x = Math.PI;
    nozzle.position.y = -ez * SCALE * 0.65;
    thGroup.add(nozzle);

    const thAnchor =
      viewMode === "fea_stress"
        ? deformedPos[geom.toolhead_node_id]
        : basePos[geom.toolhead_node_id];
    if (thAnchor) thGroup.position.copy(thAnchor);
    group.add(thGroup);
    if (viewMode === "modal") {
      stateRef.current.modalToolhead = thGroup;
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
  }, [sim, solidFea, viewMode, selectedModeIdx, feaScale, layerLimit]);

  const maxLayers = Math.max(0, (sim?.virtual_print.layer_count ?? 1) - 1);
  const activeMode = sim?.modal_fea.modes[selectedModeIdx];

  return (
    <div className="viewport-wrap">
      <div ref={containerRef} className="viewport-canvas" />

      <div className="viewport-hud-top">
        <div className="hud-pill-group">
          {(
            [
              ["assembly", "Assembly"],
              ["fea_stress", "Frame FEA"],
              ["modal", "Modal Vibration"],
              ["toolpath", "Virtual Print"],
              ["solid_part", "3D Solid Part FEA"],
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

        {sim && (
          <div className="hud-card">
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

      <div className="viewport-hud-bottom">
        {viewMode === "fea_stress" && sim && (
          <div className="hud-card">
            <span>0 MPa</span>
            <div className="legend-gradient" />
            <span>{sim.static_fea.max_von_mises_mpa.toFixed(2)} MPa</span>
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
        )}

        {viewMode === "modal" && sim && (
          <div className="hud-card">
            <span>Mode:</span>
            {sim.modal_fea.modes.slice(0, 6).map((m, idx) => (
              <button
                key={m.mode_number}
                className={`hud-pill ${selectedModeIdx === idx ? "active" : ""}`}
                style={{ padding: "4px 8px" }}
                onClick={() => onSelectModeIdx(idx)}
              >
                M{m.mode_number}: {m.frequency_hz.toFixed(1)}Hz ({m.dominant_axis})
              </button>
            ))}
          </div>
        )}

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

        {viewMode === "solid_part" && solidFea && (
          <div className="hud-card">
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

        {viewMode === "assembly" && (
          <div className="hud-card" style={{ color: "var(--muted)" }}>
            Drag to orbit • Scroll to zoom • Right-drag to pan
          </div>
        )}

        {viewMode === "modal" && activeMode && (
          <div className="hud-card">
            <span>
              Toolhead Participation XYZ:{" "}
              <strong>
                {activeMode.toolhead_participation_xyz.map((v) => `${Math.round(v * 100)}%`).join(" / ")}
              </strong>
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
