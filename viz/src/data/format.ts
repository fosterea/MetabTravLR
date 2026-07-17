/** Number formatting for edge scores shown in the tooltip and edge-details panel.
 *  C_np strengths range from ~0 to hundreds; FDRs are small (often scientific). */

/** Communication strength (C_np). Compact, magnitude-aware. */
export function formatStrength(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toPrecision(2);
}

/** FDR / p-value: fixed for readable magnitudes, scientific for the tiny ones. */
export function formatFdr(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  if (v < 0.001) return v.toExponential(1);
  return v.toFixed(3);
}
