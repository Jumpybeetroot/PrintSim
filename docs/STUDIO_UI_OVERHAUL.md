# Studio UI Overhaul (November 2026)

A record of the review and five passes that brought the web studio (`web/`) onto one visual
system. The rules that came out of it are in [`STUDIO_STYLE_GUIDE.md`](STUDIO_STYLE_GUIDE.md);
this file explains what changed and why, and what is still open.

## Why

A review of the running studio at 1440, 1024, 768 and 375px found a sound three-pane structure
and strong engineering content, but no enforced visual system and everything shown at once at
very small sizes.

| Measured in `styles.css` | Before | After |
|---|---|---|
| Distinct font sizes | 17, down to 8.5px | 6 tokens, 11px floor |
| Distinct corner radii | 15 | 3 tokens (4px, 6px, pill) plus circles and thin bars |
| Font weights | 7, mostly 700–800 | 4 (400 / 500 / 600 / 700) |
| Hex colours | 83, plus 283 `rgba()` literals | 32, almost all token definitions |
| Gradients | 58 | 7, all data (legends, hatching, 3D backdrop) |
| Form controls with an accessible name | a handful of 130 | all 130 |
| Controls under 24px tall | 53 | 0 (checkboxes are 16px inside 24px rows) |
| Layout breakpoints below 1240px | none | 1100px and 720px |

## What was done

### 1. Foundation

- **Design tokens** in the `:root` block of `styles.css`: surfaces, text, colour roles with text
  tints and channel triplets, a six-step type scale, radii, fonts.
- **Scripted normalisation** of the stylesheet onto those tokens (font sizes, radii, weights,
  a 4px spacing grid, colour families, alpha steps) and of inline `fontSize` / `fontWeight` in
  components.
- **Decorative gradients removed** from buttons, cards, banners and menus.
- **Inter self-hosted** through `@fontsource-variable/inter`. The CSS had named Inter without
  ever loading it. Buttons and inputs now inherit the font.
- **Base components restyled:** buttons (neutral default, one filled primary), inputs with a
  focus ring, tabs, cards, menus, file inputs; one global `:focus-visible` ring; reduced-motion
  respected; themed scrollbars.
- **Narrow-screen layout:** a Design / 3D View / Results switcher below 1100px; one pane at a
  time and a wrapping top bar below 720px.

Bugs fixed in this pass:

- **Deep tab clipped its content.** Cards inside the flex-column `.sidebar-scroll` were shrinking
  instead of scrolling. Fixed with `.sidebar-scroll > * { flex-shrink: 0 }`.
- **"Main limit: Main limit:"** on modal mode cards. The backend string already carries the
  prefix; the frontend now strips it before adding its own.

### 2. Declutter

- **No boxes inside boxes.** Sub-sections and disclosure rows use dividers; tiles and notes use
  fill; callouts use a left bar. Dashed dividers became solid.
- **Pills reduced to three kinds** (neutral, status, accent) defined in one place. Read-outs and
  FLEX/MASS tags became neutral.
- **Diagnostics** list critical items first and state their severity in words.
- **Truncation fixed:** cause names and Deep-tab titles wrap; long-option select pairs are
  stacked or given a 2:1 split.
- **Unit labels** standardised to `Name (unit)`.
- **Raw identifiers replaced** with names in the cut list, stock-bar nesting and diagnostics,
  through the new `web/src/utils/displayText.ts`.
- **Duplicate React key** in the modal modes card fixed.

### 3. Restructure

- **Preset and filament pinned above the tabs** (`.design-header`); they were repeated at the
  top of all four left tabs.
- **Jump bar on the Printer tab** (`.section-jump`, `DESIGN_SECTIONS`, `data-design-section`)
  with a highlight that follows scrolling.
- **Sliders show their value once:** the label read-out is hidden where a number box sits beside
  the slider.
- **"Run Print Sim" button** on the G-code box; it runs the simulation and switches to the Print
  view.

### 4. Accessibility

- **Labels linked to controls:** 98 by `htmlFor`/`id`; button groups as named `role="group"`;
  number boxes, file inputs and code boxes given `aria-label`.
- **Tabs** are real tab lists with arrow-key, Home and End support (`web/src/utils/a11y.ts`).
- **39 toggle buttons** expose `aria-pressed`; view modes are a named group; the 3D view has a
  description.
- **24px minimum target** for buttons, selects, sliders and summaries.
- **`TooltipLayer`** (`web/src/components/TooltipLayer.tsx`) shows existing `title` text on
  hover, keyboard focus and tap, and a label's `title` when its field is focused.
- **Text contrast** measured: lowest is 4.7:1 (white on the primary button); muted text is
  5.9:1 or better.

### 5. Guardrails

- [`STUDIO_STYLE_GUIDE.md`](STUDIO_STYLE_GUIDE.md): the rules.
- [`web/scripts/check-style.mjs`](../web/scripts/check-style.mjs): fails the build on raw
  colours, off-scale sizes, radii and weights, decorative gradients, unlabelled controls and
  growth in inline styles. `npm run build` now runs it first.
- `.agents/rules/web-studio.md` and `.agents/agents/studio-frontend.md` point at both.

## Decisions worth knowing

- **Input borders stay at 1.2:1** against the card, below the 3:1 guideline for control
  boundaries. Raising them would outline every field heavily; text contrast and focus rings pass.
- **Cyan and teal were merged into the accent blue** during colour consolidation.
- **Section titles are neutral, not blue**, so that blue marks only interactive or selected things.
- **The diagnostics classifier was already there.** The review said diagnostics were all amber
  regardless of severity; in fact they were classified by keyword and the reference preset simply
  had four warnings. The pass improved presentation, not classification.

## Still open

- **Inline styles in components** (395 blocks) still carry their own colours and bypass the
  tokens. The check caps the count; reducing it is future work.
- **Diagnostic severity is a keyword match** on solver text. A reliable version needs a
  `severity` field from the backend.
- **Explanations on non-interactive text** are reachable by hover and tap but not by keyboard.
- **Drawers and the Pareto modal** have not been checked for focus trapping or focus return.
- **Charts** have short descriptions at best; their data is not available as text.
- **No screen-reader run** was done; names, roles and states were checked in the DOM only.
- **Not re-opened after the later passes:** the ADXL calibration panel, the Pareto modal and the
  Compare drawer. They share the pill and container rules, so they changed too.
- **The jump bar's smooth-scroll animation** was verified with instant scrolling only.

## How it was verified

Each pass ended with `npm run build --prefix web`, `uv run pytest -q` (407 passing) and
`uv run printsim evaluate examples/corexy_voron_style.yaml`, plus a walk through the sidebar
tabs and viewport modes in the browser at desktop, tablet and phone widths. Keyboard behaviour
(tab arrow keys, focus tooltips, Escape) was tested with real key presses.
