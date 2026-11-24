/**
 * Display-only text clean-up for strings that arrive from the solver or BOM with
 * ASCII units and raw identifiers. Never use these on values that are sent back.
 */

const ACRONYMS = new Set([
  "abs", "asa", "pla", "petg", "tpu", "pa", "pa12", "cf", "cnc", "awd", "mgn", "pom", "pc", "al", "foc",
]);

/** "abs_printed" -> "ABS printed", "aluminum_6061_t6" -> "aluminum 6061 T6". */
export function humanizeId(id: string): string {
  return id
    .split("_")
    .map((w) => (ACRONYMS.has(w.toLowerCase()) || /^[a-z]\d+$/i.test(w) ? w.toUpperCase() : w))
    .join(" ");
}

/** "mm^3/s" -> "mm³/s", "mm/s^2" -> "mm/s²". */
export function prettifyUnits(text: string): string {
  return text.replace(/\^2\b/g, "²").replace(/\^3\b/g, "³");
}

/** Solver diagnostic -> readable sentence: proper unit glyphs, no quoted snake_case ids. */
export function prettifyDiagnostic(text: string): string {
  return prettifyUnits(text).replace(/'([a-z0-9]+(?:_[a-z0-9]+)+)'/g, (_, id: string) => humanizeId(id));
}

/**
 * Cut-list names arrive as "Role (Profile Name (20x20mm))". Split them so the role
 * is the row title and the profile is the subtitle.
 */
export function splitBomName(name: string, fallbackSub: string): { title: string; sub: string } {
  const i = name.indexOf(" (");
  if (i > 0 && name.endsWith(")")) {
    return { title: name.slice(0, i), sub: name.slice(i + 2, -1) };
  }
  return { title: name, sub: humanizeId(fallbackSub) };
}
