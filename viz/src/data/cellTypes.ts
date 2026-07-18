/**
 * Centralized, purely-cosmetic cell-type classification. This is the ONE place that
 * heuristically groups a cell-type name into a display family, so no string checks for
 * "T cell" leak into components (see docs/02_style_and_conventions.md). The app logic stays
 * use-case-agnostic; this only drives node color/emphasis. `background` is the harreman
 * catch-all label for non-focal cells.
 */
export type CellFamily = 'tcell' | 'other';

const BACKGROUND_LABELS = new Set(['other', 'background', 'rest']);

/**
 * Lineage markers that appear in tier labels WITHOUT the words "T cell". The finer tiers of the
 * Results/ datasets annotate subtypes directly ("CD4", "Cytotoxic CD8", "Exhausted CD4",
 * "Treg"), so a "t cell" substring test alone painted every node as background there.
 */
const TCELL_PATTERNS = [
  /\bt\s*cell\b/, // "T Cell", "CD8 T Cell", "Effector CD8 T cell"
  /\bt-cell\b/,
  /\bcd[48]\b/, // "CD4", "Cytotoxic CD8", "Exhausted CD4"
  /\btregs?\b/, // "Treg"
  /\bgamma\s*delta\b/,
];

export function classifyCellType(name: string): CellFamily {
  const n = name.trim().toLowerCase();
  if (BACKGROUND_LABELS.has(n)) return 'other';
  if (TCELL_PATTERNS.some((re) => re.test(n))) return 'tcell';
  return 'other';
}

export const isBackgroundCellType = (name: string) =>
  BACKGROUND_LABELS.has(name.trim().toLowerCase());
