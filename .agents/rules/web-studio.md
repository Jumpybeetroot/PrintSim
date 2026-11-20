---
trigger: glob
globs: web/**
---

# Web 3D Studio Rules (`web/`)

1. **Coordinate Frame Convention**:
   - Backend geometry uses CAD **Z-up** `(x, y, z)` in mm.
   - Three.js uses **Y-up**. Always convert via `toThree([x, y, z], cx, cy)` -> `((x - cx) * SCALE, z * SCALE, -(y - cy) * SCALE)` so `+X` is right, `+Z` is up, and `+Y` points toward the rear of the printer.
2. **GPU Memory Hygiene**:
   - Whenever rebuilding scene groups inside `useEffect`, explicitly call `.dispose()` on every removed mesh's `geometry` and `material` to prevent WebGL context leaks during rapid parametric slider edits.
3. **TypeScript Strictness**:
   - Keep `web/src/types.ts` strictly synchronized with the JSON shape returned by `src/printsim/api/server.py`.
   - Run `npm run build --prefix web` (`tsc && vite build`) to verify zero type or bundle errors after any frontend edit.
4. **Scientific Visualization Clarity**:
   - Preserve clear legends and scales for FEA stress heatmaps, deflection exaggeration multipliers, modal mode frequency labels, and G-code defect colors.
