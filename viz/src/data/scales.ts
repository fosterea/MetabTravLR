/**
 * Visual encodings for edges. Magnitudes (harreman C_np) span orders of magnitude,
 * so edge width uses a LOG scale (decision A6). To keep the widths *distinguishable*
 * we normalize between the view's log-min and log-max (not 0..max) so the full px
 * range is used across the actual spread — otherwise everything hugs the floor and
 * the differences read as tiny (see docs/02_style_and_conventions.md).
 */
import type { EntityEdge } from './types';

export const EDGE_WIDTH_PX = { min: 2, max: 22 } as const;

export interface EdgeWidthScale {
  (value: number): number;
  domainMax: number;
  domainMin: number;
}

/**
 * Build a log width-scale over the strengths present in the current view.
 * Maps the smallest visible strength → ~min px and the largest → max px, spreading
 * contrast across the whole range. A single (or all-equal) edge maps to max so a
 * lone interface still reads as substantial rather than a hairline.
 */
export function makeEdgeWidthScale(
  edges: EntityEdge[],
  pick: (e: EntityEdge) => number = (e) => e.scores.C_np,
): EdgeWidthScale {
  const strengths = edges.map((e) => Math.max(0, pick(e) || 0));
  const domainMax = strengths.reduce((m, v) => Math.max(m, v), 0);
  const domainMin = strengths.length ? strengths.reduce((m, v) => Math.min(m, v), Infinity) : 0;
  const lo = Math.log1p(domainMin);
  const hi = Math.log1p(domainMax);
  const span = hi - lo;
  const scale = ((value: number): number => {
    const v = Math.max(0, value || 0);
    // No spread (one edge, or all identical) → render at max so it's clearly visible.
    const t = span > 0 ? (Math.log1p(v) - lo) / span : 1;
    return EDGE_WIDTH_PX.min + (EDGE_WIDTH_PX.max - EDGE_WIDTH_PX.min) * clamp01(t);
  }) as EdgeWidthScale;
  scale.domainMax = domainMax;
  scale.domainMin = domainMin;
  return scale;
}

const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);

/** True when an edge is a within-cell-type (diagonal / self-loop) interface. */
export const isSelfEdge = (e: EntityEdge) => e.source === e.target;
