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
5. **Visual System & Style Guide** (`docs/STUDIO_STYLE_GUIDE.md`):
   - Read `docs/STUDIO_STYLE_GUIDE.md` before changing anything in `web/src/`. It defines the design tokens, the container and pill patterns, label/unit wording, and the accessibility requirements.
   - Use tokens only: no raw colours, no font sizes outside `--fs-*` (11px floor), no radii outside `--radius-sm` / `--radius` / `--radius-pill`, no weights outside 400/500/600/700, no decorative gradients, no new inline `style={{}}` for static styling.
   - Colour carries meaning (accent = interactive/selected; green/amber/red = status; purple = beta). Pills default to neutral. No bordered boxes inside a `.section-card`.
   - Every control needs an accessible name (`<label htmlFor>` + `id`, a wrapping label, or `aria-label`); toggles set `aria-pressed`; explanations go in `title` (the shared `TooltipLayer` handles hover, focus and tap).
   - `npm run build --prefix web` runs `web/scripts/check-style.mjs` first and fails on violations. Never weaken or bypass the check to land a change; if a rule is wrong, update the guide and the script together and say why.
   - After a visual change, look at the result in the browser at 1440px and 375px wide before declaring it done.
