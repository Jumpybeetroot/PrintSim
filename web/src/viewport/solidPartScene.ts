import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SolidFEAResponse, TetFEAResponse, TopologyOptResponse } from "../types";
import { stressColor } from "./colormaps";
import { frameCameraToBox } from "./geometryHelpers";

export interface SolidPartCameraState {
  camera?: THREE.PerspectiveCamera;
  controls?: OrbitControls;
  lastFramedKey?: string;
}

export interface BuildSolidPartFeaParams {
  group: THREE.Group;
  showTetPart: boolean;
  tetFea: TetFEAResponse | null;
  solidFea: SolidFEAResponse | null;
  feaScale: number;
  cameraState: SolidPartCameraState;
}

export function buildSolidPartFeaScene({
  group,
  showTetPart,
  tetFea,
  solidFea,
  feaScale,
  cameraState,
}: BuildSolidPartFeaParams): void {
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
        if (cameraState.lastFramedKey !== frameKey) {
          const partBox = new THREE.Box3(
            new THREE.Vector3((minX - pcx) * partScale, 0.4, -(maxY - pcy) * partScale),
            new THREE.Vector3(
              (maxX - pcx) * partScale,
              (maxZ - minZ) * partScale + 0.4,
              -(minY - pcy) * partScale
            )
          );
          frameCameraToBox(cameraState.camera, cameraState.controls, partBox);
          cameraState.lastFramedKey = frameKey;
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
      if (cameraState.lastFramedKey !== frameKey) {
        const halfBox = boxSize * 0.5;
        const partBox = new THREE.Box3(
          new THREE.Vector3((minX - pcx) * partScale - halfBox, 0.4, -(maxY - pcy) * partScale - halfBox),
          new THREE.Vector3(
            (maxX - pcx) * partScale + halfBox,
            (maxZ - minZ) * partScale + boxSize + 0.4,
            -(minY - pcy) * partScale + halfBox
          )
        );
        frameCameraToBox(cameraState.camera, cameraState.controls, partBox);
        cameraState.lastFramedKey = frameKey;
      }
      return;
}

export interface BuildTopologyOptParams {
  group: THREE.Group;
  topoResult: TopologyOptResponse | null;
  feaScale: number;
  showCarvedGhost: boolean;
}

export function buildTopologyOptScene({
  group,
  topoResult,
  feaScale,
  showCarvedGhost,
}: BuildTopologyOptParams): void {
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
