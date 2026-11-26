# PrintSim Studio Style Guide

This is the source of truth for how the web studio (`web/`) looks and behaves. Read it before
changing anything under `web/src/`. It applies to people and to AI agents equally.

- **Tokens and base styles:** [`web/src/styles.css`](../web/src/styles.css) importing ordered partials under [`web/src/styles/`](../web/src/styles/) (`tokens.css` holds the `:root` block; `layout.css`, `analysis.css`, `viewport.css`, `panels.css`, `modals.css` hold component rules)
- **Automated check:** [`web/scripts/check-style.mjs`](../web/scripts/check-style.mjs), run by `npm run build --prefix web` (scans every `web/src/**/*.css` and `web/src/**/*.tsx` file)
- **History of how it got here:** [`STUDIO_UI_OVERHAUL.md`](STUDIO_UI_OVERHAUL.md)

The check enforces the mechanical rules (section 11). The rest of this guide is judgement that a
script cannot check, so it relies on you.

---

## 1. Principles

1. **One system, no one-offs.** Every size, colour, radius and weight comes from a token. If the
   value you want is not a token, use the nearest one. Do not add a new value for one element.
2. **Colour means something.** Blue is interactive or selected. Green, amber and red are status.
   Purple is beta/generative. Everything else is neutral. Never use colour as decoration.
3. **No boxes inside boxes.** The card is the only bordered container in a sidebar. Inside a card,
   group with dividers, fill or a left bar (section 5).
4. **Say it once.** A value appears in one place. A control appears in one place. Do not repeat
   a read-out next to the input that already shows it.
5. **Readable by default.** Nothing below 11px. Nothing truncated that the user needs to read.
   Names, not identifiers. Real unit glyphs.
6. **Usable without a mouse.** Every control has a name, a visible focus ring and a 24px target.

---

## 2. Tokens

All defined in the `:root` block of `web/src/styles/tokens.css` (imported first by `styles.css`). Use the variable, never the literal.

### Surfaces and text

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0b0f16` | Page background |
| `--panel` | `#121822` | Sidebars, top bar, menus |
| `--panel-alt` | `#171e2b` | Cards, default buttons |
| `--panel-hover` | `#1e2736` | Hover and open states, tooltip |
| `--inset` | `#0d121b` | Inputs, tiles, neutral pills |
| `--border` | `#252f42` | Borders and dividers |
| `--border-strong` | `#37445c` | Hover borders, menu and tooltip outlines |
| `--text` | `#e6ebf4` | Primary text |
| `--text-2` | `#c2cddd` | Section titles, secondary text |
| `--muted` | `#8c9bb3` | Labels, helper text |
| `--ink` | `#08101c` | Dark text on a bright fill (rare) |

### Colour roles

| Role | Solid | Text tint | Meaning |
|---|---|---|---|
| Accent | `--accent` | `--accent-text` | Interactive, selected, live |
| Primary fill | `--accent-strong` | (white text) | The one filled button per area |
| Good | `--green` | `--green-text` | Passing, improved, recommended |
| Caution | `--amber` | `--amber-text` | Warning, limiting, needs attention |
| Bad | `--red` | `--red-text` | Critical, failing, worse |
| Beta | `--purple` | `--purple-text` | Beta and generative features only |

Tinted backgrounds and borders use the channel triplets: `rgb(var(--accent-rgb) / 0.12)`.
Standard strengths are **0.12** for a fill and **0.4** for a border. Available triplets:
`--accent-rgb`, `--green-rgb`, `--amber-rgb`, `--red-rgb`, `--purple-rgb`, `--muted-rgb`,
`--border-rgb`, `--panel-rgb`, `--inset-rgb`.

Use the `*-text` tint for coloured **text** on dark surfaces. The solid colour is for bars, dots
and borders.

### Type

| Token | Size | Use |
|---|---|---|
| `--fs-xs` | 11px | Labels, pills, helper text, table cells. **The floor.** |
| `--fs-sm` | 12px | Inputs, buttons, body text in panels |
| `--fs-md` | 13px | Base size, brand subtitle |
| `--fs-lg` | 15px | KPI values, brand |
| `--fs-xl` | 20px | Large figures |
| `--fs-2xl` | 32px | The composite score only |

- **Weights:** 400 body, 500 labels and buttons, 600 titles and emphasis, 700 key figures. Nothing else.
- **Fonts:** `var(--font)` (Inter, self-hosted through `@fontsource-variable/inter`) and
  `var(--font-mono)` for code. Buttons and inputs inherit the font; do not set `font-family` elsewhere.
- **Uppercase** is for section titles, table headers and tile labels (`.kpi-label`) only, with
  `letter-spacing: 0.04–0.06em`. Never uppercase a value or a pill.
- Numbers that change or line up in columns use `font-variant-numeric: tabular-nums`.

### Shape and spacing

- **Radius:** `--radius-sm` (4px) for small inline elements, `--radius` (6px) for everything else,
  `--radius-pill` for pills and chips. `50%` for dots. 1–2px only for thin bars.
- **Spacing:** 4px grid. Use 4, 8, 12, 16. Use 2 and 6 only inside dense rows. Card padding is
  12px, gap between cards 12px, gap between fields in a card 8px.

---

## 3. Layout

```
top bar (48px)        brand | Optimize, Export, Pin/Compare, Save/Load, [Re-Simulate]
pane switch           only below 1100px: Design | 3D View | Results
workspace             left sidebar 330px | 3D viewport | right sidebar 360px
```

- **Top bar:** neutral `.btn` buttons and exactly one `.btn.primary` (Re-Simulate). Menus are
  plain lists (`.topbar-menu-item`); do not tint individual items.
- **Left sidebar:** the pinned `.design-header` (Printer, Filament, auto-save) sits **above** the
  tabs because it applies to every tab. Do not put it back inside a tab.
- **Printer tab:** `.section-jump` is a sticky jump bar. Each top-level section card carries
  `data-design-section="<id>"`, and the list of sections is `DESIGN_SECTIONS` in
  `DesignPanel.tsx`. **Adding a section means adding both.**
- **Sidebars scroll, cards do not shrink.** `.sidebar-scroll > *` has `flex-shrink: 0`. Never
  remove it: without it, cards with `overflow: hidden` are squashed and clipped.
- **Narrow screens:** below 1100px the workspace shows one sidebar beside the viewport, chosen by
  `.pane-switch`; below 720px it shows one pane at a time. Any new top-level pane must work in
  that scheme. Check 1440, 1024, 768 and 375px wide before finishing.

---

## 4. Containers

### The card

`.section-card` is the only bordered container in a sidebar: `--panel-alt` fill, `--border`
outline, 12px padding. Its heading is `.section-title` (11px, uppercase, `--text-2`), optionally
with one neutral pill on the right.

Do not give a card a coloured border to make it stand out. The exception is `.impact-card`,
which appears only after a change and is highlighted on purpose.

### Inside a card: no boxes within boxes

| You need | Use | Looks like |
|---|---|---|
| A titled sub-section | `.sub-block` + `.sub-block-title` | Top divider, no box |
| Collapsible advanced settings | `<details className="sub-accordion">` | Top divider, chevron, summary tag on the right |
| A metric tile | `.kpi-box` in `.kpi-grid` / `.kpi-grid-3` | `--inset` fill, no border |
| A helper note | a `*-note` class | Fill only, no border |
| A list of ranked items | rows separated by `border-top: 1px solid var(--border)` | Like `.top-cause-row` |
| A status callout | 3px left bar in the status colour + a faint tint | Like `.diag-item`, `.modal-mode-item` |
| An Advanced Mode override strip | `.model-override-strip` (fill-only `--inset` + 2px left bar, `--accent` when `.overridden`) | Hidden unless Advanced Mode is ON; expandable compact inputs + Reset |

Dividers are solid `1px solid var(--border)`. **Dashed lines mean "drop a file here" or "nothing
here yet"** and nothing else.

Gradients are for **data** only: colour-scale legends, hatching, chart swatches and the 3D
backdrop. Never on a button, card, pill or banner.

---

## 5. Controls

### Buttons

| Class | Use |
|---|---|
| `.btn` | Default. Neutral fill, neutral border. |
| `.btn.primary` | The single main action of an area (Re-Simulate, Run Part FEA, Apply YAML). One per area. |
| `.btn.ai` | Purple-tinted, for generative actions only. |
| `.btn.pareto`, `.btn.deep-verify` | Accent-tinted secondary actions. |
| `.chip-btn` (+ `.active`) | Small filter or jump chips. |

Hover changes the fill to `--panel-hover` and the border to `--border-strong`. Do not invent
hover colours. Disabled is `opacity: 0.5`.

A toggle button must set `aria-pressed`, not only an `active` class.

### Fields

```tsx
<div className="field-row">
  <label htmlFor="dp-belt-type-7">Belt Type</label>
  <select id="dp-belt-type-7" …>
</div>
```

- One field per `.field-row`. Put fields side by side with `.field-grid-2` or `.field-grid-3`.
- **Long option names:** if either select in a pair has long options, use `.field-grid-2.stack`
  (full width each) or `.wide-first` / `.wide-last` (2:1). Do not leave a select truncated.
- Inputs, selects and text areas use the base styles. Do not restyle them per component.
- **Sliders:** a `.slider-row` with the range input and, where precision matters, a
  `.tension-num-input` number box beside it. When the number box is present, do not also print the
  value in the label; the label may carry context ("stock", "lever arm") or a status pill.
- **Checkboxes:** `<label className="checkbox-row"><input type="checkbox" /> Text</label>`. The
  text sits next to the box.
- **File inputs** use the native control; the base styles dress it as a button. Give it an
  `aria-label`.

### Labels and units

- Format is **`Name (unit)`**: `Width (mm)`, `Belt Tension (N)`, `Heater (W)`. Not `Width mm`,
  not `Heater W`.
- Sentence-style title case, no trailing colon.
- A label describes the field. Explanations go in a `title` (section 8) or a hint line, not in the label.

---

## 6. Pills and tags

Defined once, under "Pills & tags" in `styles.css`. Every small tag is one of three kinds:

| Kind | Looks like | Use for | Classes |
|---|---|---|---|
| **Neutral** | `--inset` fill, `--border`, `--text-2` | A read-out or label: `X 450g / Y 1050g`, `3mm Acrylic`, `FLEX` | `.badge-pill`, `.sub-block-tag`, `.sub-accordion-tag`, `.tension-hz-pill`, `.mount-pill`, `.sens-badge`, `.cause-kind-tag`, `.deep-accordion-badge`, `.modal-severity-pill` |
| **Status** | Tint + text tint in green / amber / red | What a value **means**: ok, warning, limiting | add `.ok`, `.warn`, `.bad` / `.danger` to the above |
| **Accent** | Accent tint | Selected or live; purple for beta | `.saved-active-badge`, `.changed-dot-label`, `.badge-pill.info`, `.mount-pill.info`, `.modal-severity-pill.info`, `.beta-pill` |

Rules:

- **Default to neutral.** A pill is coloured only when its colour tells the user something.
- One shape: pill radius, 11px, weight 500, `1px 8px` padding, never uppercase, never wraps.
- **To add a pill, reuse an existing class.** If you add a new pill class, add it to the shared
  selector list in the "Pills & tags" section and do not give it its own colours.
- Category tags whose colour matches a chart legend (`.cause-cat-pill`) keep that colour: it is
  data, not decoration.

---

## 7. Text and data

- **Names, not identifiers.** Never show `extrusion_2020` or `'abs_printed'`. Use the catalog
  label, or the helpers in [`web/src/utils/displayText.ts`](../web/src/utils/displayText.ts):
  `humanizeId`, `prettifyUnits`, `prettifyDiagnostic`, `splitBomName`.
- **Real unit glyphs:** `mm³/s`, `mm/s²`, `µm`, `°C`, `×`. Not `mm^3/s`.
- **Do not truncate what the user needs.** Let names wrap; let the less important neighbour
  (a summary tag) give way. Truncate with an ellipsis only as a last resort, and then put the
  full text in a `title`.
- **Severity is written, not only coloured.** Diagnostics carry a "Critical" / "Warning" / "Note"
  label, and critical items come first.
- A solver string may already include a prefix such as "Main limit:". Check before adding one.

---

## 8. Accessibility

These are requirements, not nice-to-haves.

- **Every control has an accessible name.** A `<label htmlFor>` pointing at the control's `id`;
  or the label wraps the control; or an `aria-label`. A label that heads a group of buttons gets
  an `id`, and the group gets `role="group" aria-labelledby`.
- **Ids are unique.** Use the `dp-` / `df-` / `db-` prefix of the file plus a descriptive slug.
  Inside a `.map()`, build the id from the item's key.
- **Tabs** are `role="tablist"` / `role="tab"` with `aria-selected`, a roving `tabIndex`, and
  `onTabListKeyDown` from [`web/src/utils/a11y.ts`](../web/src/utils/a11y.ts).
- **Toggle buttons** set `aria-pressed`. Menu triggers set `aria-haspopup` and `aria-expanded`.
- **Targets are at least 24×24px.** The base styles guarantee this for buttons, selects and
  sliders; do not override `min-height` downward.
- **Focus** uses the single global `:focus-visible` ring. Never set `outline: none` without an
  equivalent replacement.
- **Explanations go in `title`.** The shared `TooltipLayer` shows any `title` on hover, on
  keyboard focus and on tap, and shows a label's `title` when its field is focused. Do not build
  another tooltip.
- **Motion** respects `prefers-reduced-motion` through the global rule. New animations must not
  bypass it.
- **Text contrast** is at least 4.5:1. `--muted` on a card is 5.9:1; do not go dimmer than
  `--muted` for text.
- **Known and accepted:** input borders are 1.2:1 against the card, below the 3:1 guideline for
  control boundaries. This was a deliberate choice for a quieter look. Do not "fix" it in passing.

---

## 9. The 3D viewport

- HUD cards and pill groups float over the canvas with `rgb(var(--inset-rgb) / 0.95)` fills.
- View-mode buttons are `.hud-pill` in a `role="group"`; the active one uses `--accent-strong`.
- Every colour scale has a legend with its units and range. Legends are the one place gradients
  belong.
- Three.js materials take hex numbers, not CSS variables. Those colours live in the viewport
  code and are outside this guide; keep them consistent with the colour roles where they carry
  the same meaning (red for limiting, green for ok).
- Coordinate, disposal and type rules are in `.agents/rules/web-studio.md`.

---

## 10. Where things go

| You are adding | Put it in |
|---|---|
| A colour, size or radius | Nowhere new. Use a token. |
| Styling for a component | A class in `styles.css`, next to related rules. Not an inline `style`. |
| A new kind of pill, card or button | Extend the shared rule in `styles.css`; update this guide. |
| Display clean-up of solver text | `web/src/utils/displayText.ts` |
| Keyboard or ARIA helpers | `web/src/utils/a11y.ts` |
| A new Printer-tab section | A `.section-card` with `data-design-section`, plus an entry in `DESIGN_SECTIONS` |

Inline `style={{…}}` is allowed only for values computed at runtime (a bar's width, a colour
from data). The number of inline style blocks is capped by the check and may only go down.

---

## 11. What the check enforces

`npm run check:style --prefix web` (also the first step of `npm run build`). It fails the build on:

| Rule | Scope |
|---|---|
| `font-size` is a `--fs-*` token | `styles.css` |
| `font-weight` is 400, 500, 600 or 700 | `styles.css` and inline styles |
| `border-radius` is a radius token, `50%`, or ≤2px | `styles.css` |
| `font-family` is `var(--font)`, `var(--font-mono)` or `inherit` | `styles.css` |
| No raw hex or `rgb()` colours outside `:root` (white and black excepted) | `styles.css` |
| No gradients outside data selectors | `styles.css` |
| No dashed borders outside drop-target and empty-state selectors | `styles.css` |
| Inline `fontSize` is at least 11 | `*.tsx` |
| Every `<label>` is tied to a control or a group | `*.tsx` |
| Every `<select>` and `<textarea>` has an `id` or `aria-label` | `*.tsx` |
| Inline `style={{}}` blocks do not exceed the cap | `*.tsx` |

Data selectors and the inline-style cap are listed at the top of the script. If a rule blocks a
change you believe is right, change this guide and the script together, in the same commit, and
say why. Do not add an exception to get one change through.

**The check cannot see:** whether a colour is used with the right meaning, nested boxes,
truncation, wording, layout at narrow widths, or whether focus order makes sense. Those are the
checklist below.

---

## 12. Checklist before you finish a UI change

1. `npm run build --prefix web` passes (style check, types, bundle).
2. Looked at it in the browser at **1440px and 375px** wide, and at 768px if the layout changed.
3. No new colour, size, radius or weight; no new inline styles.
4. Nothing new is boxed inside a card; no new coloured pill without a meaning.
5. Nothing important is truncated; no raw identifiers or ASCII units on screen.
6. Tabbed through the new controls: each has a name, a visible focus ring and a tooltip if it
   has a `title`.
7. If you added a Printer-tab section, it is in the jump bar.
8. If you changed a rule, this guide and `check-style.mjs` are updated in the same commit.
9. **Flex-direction & header-wrapping safety** (see `docs/CAD_AND_FEA_PITFALLS.md` §5): never use `flex: 1 0 100%` or `flex-basis: 100%` inside a `flex-direction: column` container (where `flex-basis` controls **height**, clipping sibling rows), never nest `.field-row` inside `.field-row`, and ensure every sidebar header pairing a label with status pills has `flex-wrap: wrap`.
