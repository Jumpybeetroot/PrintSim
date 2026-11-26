#!/usr/bin/env node
/**
 * Studio style check. Runs before every build (`npm run build`) and on demand
 * (`npm run check:style`). It enforces the rules in docs/STUDIO_STYLE_GUIDE.md that
 * can be checked mechanically, so the visual system cannot drift one edit at a time.
 *
 * No dependencies: plain Node, regex over src/**\/*.css and src/**\/*.tsx.
 * If a rule blocks a change you believe is right, change the guide and this file
 * together in the same commit rather than working around the check.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "src");
const errors = [];
const fail = (file, line, message) => errors.push(`${relative(root, file)}:${line}  ${message}`);

// --------------------------------------------------------------------------
// Allowances. Keep these lists short; every entry is a deliberate exception.
// --------------------------------------------------------------------------

/** Selectors that draw data (colour scales, hatching, chart swatches, the 3D backdrop). */
const DATA_COLOUR_SELECTORS = /legend-gradient|nesting-waste|psd-swatch|adxl-swatch|compare-swatch|viewport-wrap/;
/** Dashed borders mean "drop target" or "empty state" only. */
const DASHED_SELECTORS = /drop|empty|swatch/;
const ALLOWED_WEIGHTS = new Set(["400", "500", "600", "700"]);
const ALLOWED_RADII = /^(0|1px|2px|50%|var\(--radius(-sm|-pill)?\))$/;

/**
 * Ratchet: inline `style={{ ... }}` blocks in components. New styling belongs in
 * styles.css. This number may only go down; lower it when you remove some.
 */
const MAX_INLINE_STYLE_BLOCKS = 395;

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
const walkExt = (dir, ext) =>
  readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walkExt(path, ext) : path.endsWith(ext) ? [path] : [];
  });

// --------------------------------------------------------------------------
// Stylesheets (*.css under src/)
// --------------------------------------------------------------------------
const cssFiles = walkExt(src, ".css");
for (const cssPath of cssFiles) {
  const css = readFileSync(cssPath, "utf8");
  const lineOf = (index) => css.slice(0, index).split("\n").length;

  for (const rule of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    const selector = rule[1].replace(/\/\*[\s\S]*?\*\//g, "").trim().replace(/\s+/g, " ");
    const body = rule[2];
    const bodyStart = rule.index + rule[1].length + 1;
    if (selector === ":root") continue; // token definitions live here
    const dataColours = DATA_COLOUR_SELECTORS.test(selector);

    for (const decl of body.matchAll(/([a-z-]+)\s*:\s*([^;]+);/g)) {
      const [, prop, rawValue] = decl;
      const value = rawValue.trim();
      const line = lineOf(bodyStart + decl.index);
      const at = `${selector} { ${prop} }`;

      if (prop === "font-size" && !/^(var\(--fs-(xs|sm|md|lg|xl|2xl)\)|inherit)$/.test(value)) {
        fail(cssPath, line, `${at}: font-size must be a --fs-* token, got "${value}"`);
      }
      if (prop === "font-weight" && !ALLOWED_WEIGHTS.has(value)) {
        fail(cssPath, line, `${at}: font-weight must be 400, 500, 600 or 700, got "${value}"`);
      }
      if (/^border(-[a-z]+){0,2}-radius$/.test(prop)) {
        for (const part of value.split(/\s+/)) {
          if (!ALLOWED_RADII.test(part)) {
            fail(cssPath, line, `${at}: radius must be --radius-sm, --radius, --radius-pill, 50% or <=2px, got "${part}"`);
          }
        }
      }
      if (prop === "font-family" && !/^(var\(--font(-mono)?\)|inherit)$/.test(value)) {
        fail(cssPath, line, `${at}: font-family must be var(--font), var(--font-mono) or inherit`);
      }
      if (!dataColours) {
        if (/#[0-9a-fA-F]{3,8}\b/.test(value)) {
          fail(cssPath, line, `${at}: raw hex colour "${value.match(/#[0-9a-fA-F]{3,8}/)[0]}"; use a colour token`);
        }
        for (const fn of value.matchAll(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/g)) {
          const channels = fn.slice(1, 4).join(",");
          if (channels !== "255,255,255" && channels !== "0,0,0") {
            fail(cssPath, line, `${at}: raw rgb(${channels}); use rgb(var(--<role>-rgb) / <alpha>)`);
          }
        }
        if (/gradient\(/.test(value)) {
          fail(cssPath, line, `${at}: gradients are for data only (legends, hatching); use a flat token colour`);
        }
      }
      if (/\bdashed\b/.test(value) && !DASHED_SELECTORS.test(selector)) {
        fail(cssPath, line, `${at}: dashed borders are reserved for drop targets and empty states`);
      }
    }
  }
}

// --------------------------------------------------------------------------
// Components (*.tsx)
// --------------------------------------------------------------------------
let inlineStyleBlocks = 0;
for (const file of walkExt(src, ".tsx")) {
  const text = readFileSync(file, "utf8");
  const lineAt = (index) => text.slice(0, index).split("\n").length;
  inlineStyleBlocks += (text.match(/style=\{\{/g) || []).length;

  for (const m of text.matchAll(/fontSize:\s*([\d.]+)(?=[,\s}])/g)) {
    if (Number(m[1]) < 11) fail(file, lineAt(m.index), `inline fontSize ${m[1]}: the floor is 11px`);
  }
  for (const m of text.matchAll(/fontWeight:\s*(\d+)(?=[,\s}])/g)) {
    if (!ALLOWED_WEIGHTS.has(m[1])) fail(file, lineAt(m.index), `inline fontWeight ${m[1]}: use 400, 500, 600 or 700`);
  }

  // Every <label> must be tied to something: htmlFor (a control), id (heads a
  // role="group"), or it wraps its control.
  for (const m of text.matchAll(/<label\b([^>]*)>/g)) {
    if (/\bhtmlFor=|\bid=/.test(m[1])) continue;
    const end = text.indexOf("</label>", m.index);
    const inner = end === -1 ? "" : text.slice(m.index, end);
    if (/<(input|select|textarea)\b/.test(inner)) continue;
    fail(file, lineAt(m.index), "<label> is not linked to a control: add htmlFor + id, or wrap the control");
  }

  // Controls need an accessible name when no label points at them.
  for (const m of text.matchAll(/<(select|textarea)\b([^>]*)/g)) {
    if (/\b(id|aria-label|aria-labelledby)=/.test(m[2])) continue;
    const before = text.slice(Math.max(0, m.index - 400), m.index);
    if (/<label\b[^>]*>(?:(?!<\/label>)[\s\S])*$/.test(before)) continue; // wrapped by a label
    fail(file, lineAt(m.index), `<${m[1]}> has no id (for a label) and no aria-label`);
  }
}

if (inlineStyleBlocks > MAX_INLINE_STYLE_BLOCKS) {
  errors.push(
    `src/**/*.tsx  ${inlineStyleBlocks} inline style={{}} blocks, limit is ${MAX_INLINE_STYLE_BLOCKS}. ` +
      "Put new styling in styles.css as a class instead of adding inline styles."
  );
}

// --------------------------------------------------------------------------
if (errors.length) {
  console.error(`\nStudio style check failed (${errors.length}). See docs/STUDIO_STYLE_GUIDE.md.\n`);
  for (const e of errors) console.error("  " + e);
  console.error("");
  process.exit(1);
}
const slack = MAX_INLINE_STYLE_BLOCKS - inlineStyleBlocks;
console.log(
  `Studio style check passed (${cssFiles.length} CSS files, ${inlineStyleBlocks} inline style blocks` +
    (slack > 0 ? `; limit can be lowered to ${inlineStyleBlocks}` : "") +
    ")."
);
