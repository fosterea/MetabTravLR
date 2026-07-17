/**
 * Centralized, purely-cosmetic cell-type classification. This is the ONE place that
 * heuristically groups a cell-type name into a display family, so no string checks for
 * "T cell" leak into components (see docs/02_style_and_conventions.md). The app logic stays
 * use-case-agnostic; this only drives node color/emphasis. `background` is the harreman
 * catch-all label for non-focal cells.
 */
export type CellFamily = 'tcell' | 'other';

const BACKGROUND_LABELS = new Set(['other', 'background', 'rest']);

export function classifyCellType(name: string): CellFamily {
  const n = name.trim().toLowerCase();
  if (BACKGROUND_LABELS.has(n)) return 'other';
  // T-cell lineage across tiers: "T Cell", "CD8 T Cell", "Effector CD8 T cell", "Gamma delta T cell", ...
  if (/\bt\s*cell\b/.test(n) || /\bt-cell\b/.test(n)) return 'tcell';
  return 'other';
}

export const isBackgroundCellType = (name: string) =>
  BACKGROUND_LABELS.has(name.trim().toLowerCase());
