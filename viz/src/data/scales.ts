/**
 * Visual encodings for edges. Magnitudes (harreman C_np) span orders of magnitude,
 * so edge width uses a LOG scale normalized to the current view's max, clamped to a
 * px range (see docs/02_style_and_conventions.md, decision A6).
 */
import type { EntityEdge } from './types';

export const EDGE_WIDTH_PX = { min: 1.5, max: 14 } as const;

export interface EdgeWidthScale {
  (value: number): number;
  domainMax: number;
}

/** Build a log width-scale over the strengths present in the current view. */
export function makeEdgeWidthScale(
  edges: EntityEdge[],
  pick: (e: EntityEdge) => number = (e) => e.scores.C_np,
): EdgeWidthScale {
  const domainMax = edges.reduce((m, e) => Math.max(m, Math.max(0, pick(e) || 0)), 0);
  const denom = Math.log1p(domainMax) || 1;
  const scale = ((value: number): number => {
    const v = Math.max(0, value || 0);
    const t = Math.log1p(v) / denom; // 0..1
    return EDGE_WIDTH_PX.min + (EDGE_WIDTH_PX.max - EDGE_WIDTH_PX.min) * clamp01(t);
  }) as EdgeWidthScale;
  scale.domainMax = domainMax;
  return scale;
}

const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);

/** True when an edge is a within-cell-type (diagonal / self-loop) interface. */
export const isSelfEdge = (e: EntityEdge) => e.source === e.target;
