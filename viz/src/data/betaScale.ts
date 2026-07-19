/**
 * Encoding + lookup helpers for the SpaceTravLR coefficients (`BetaRow`).
 *
 * The problem this file exists to solve: the betas span ~6 orders of magnitude (|mean| runs from
 * exact 0 through ~1e-8 up to ~4.5e-3), and they are SIGNED. A linear scale renders all but a
 * handful of cells as an indistinguishable floor. A naive signed log — `sign(x) * log10(|x|)` —
 * is worse than useless: for x = -1e-7 it yields -(-7) = +7, i.e. a tiny NEGATIVE coefficient
 * becomes a large POSITIVE one. The sign is the biological claim, so an encoding that can flip
 * or inflate it is a correctness bug, not a styling choice.
 *
 * What we do instead:
 *   - MAGNITUDE and SIGN are encoded on separate channels. `norm()` returns a magnitude in
 *     [0, 1] and the sign is carried alongside, never folded into the number.
 *   - The magnitude is a log ramp anchored to a floor a FIXED number of decades below the
 *     strongest coefficient in view, so the ramp is monotonic in |x| and comparable across the
 *     whole view (all cell types share one scale — that is what makes cell types comparable).
 *   - Anything at or below that floor returns 0 and is reported `negligible`. It gets NO
 *     pedestal and no color: a near-zero coefficient must read as *nothing*, not as a small
 *     real effect. This is the deliberate difference from a min-anchored scale, which would give
 *     the view's smallest value a visible floor and make "negligible" look like "weak but there".
 */
import type { BetaRow } from './types';

/**
 * How many orders of magnitude the color ramp spans below the strongest |beta| in view.
 * Below this the coefficient is treated as negligible. Four decades is already a very wide
 * perceptual range; widening it just makes noise look like signal.
 */
export const BETA_DECADES = 4;

/**
 * Order-INDEPENDENT key into `BetaBundle.byPair`. Must match `betaKey` in `scripts/ingest.mjs`.
 * Gene-pair entity ids preserve the network's arbitrary order, so they cannot be used directly.
 */
export const betaKey = (g1: string, g2: string) => (g1 <= g2 ? `${g1}__${g2}` : `${g2}__${g1}`);

export interface BetaScale {
  /** Magnitude in [0, 1]. Always non-negative — the sign travels separately. */
  norm: (v: number | null) => number;
  /** True when |v| is at/below the floor (or v is 0/null): report as ≈0, do not color it. */
  negligible: (v: number | null) => boolean;
  /** Largest |beta| in the view — the top of the ramp. 0 when the view is empty/all-zero. */
  max: number;
  /** The floor, i.e. `max / 10^BETA_DECADES`. Anything under this reads as ≈0. */
  floor: number;
}

/**
 * Build a scale over every value shown in the current view, so cell types and gene pairs are
 * directly comparable. Pass ALL displayed values, not one block's worth.
 */
export function makeBetaScale(values: (number | null)[]): BetaScale {
  let max = 0;
  for (const v of values) {
    if (v == null) continue;
    const a = Math.abs(v);
    if (a > max) max = a;
  }
  const floor = max / 10 ** BETA_DECADES;
  const span = Math.log10(max) - Math.log10(floor); // === BETA_DECADES when max > 0

  const negligible = (v: number | null) => v == null || max === 0 || Math.abs(v) <= floor;

  const norm = (v: number | null) => {
    if (negligible(v)) return 0;
    const t = (Math.log10(Math.abs(v ?? 0)) - Math.log10(floor)) / span;
    // Clamp defensively; `t` is in (0, 1] by construction given the negligible() guard above.
    return Math.min(1, Math.max(0, t));
  };

  return { norm, negligible, max, floor };
}

/**
 * Compact fixed-width rendering of a coefficient, e.g. `-5.8e-6`, `+4.5e-3`, `≈0`.
 * The number is printed in every heatmap cell, so the color never has to carry the value alone.
 */
export function formatBeta(v: number | null, scale?: BetaScale): string {
  if (v == null) return '—';
  if (v === 0) return '0';
  if (scale?.negligible(v)) return '≈0';
  const [mant, exp] = v.toExponential(1).split('e');
  const sign = v > 0 ? '+' : '';
  return `${sign}${mant}e${Number(exp)}`;
}

/**
 * Unsigned rendering, for thresholds and scale ends where the sign is stated separately
 * (`−4.5e-3 ▮▯▮ +4.5e-3`) or meaningless (the ≈0 floor is a magnitude, not a value).
 */
export function formatMagnitude(v: number): string {
  if (v === 0) return '0';
  const [mant, exp] = Math.abs(v).toExponential(1).split('e');
  return `${mant}e${Number(exp)}`;
}

/** Full-precision label for a row's tooltip. */
export function betaTooltip(r: BetaRow): string {
  const dec = (v: number | null) => (v == null ? '—' : v.toExponential(4));
  return [
    `${r.env} (environment) → ${r.cell} (cell)`,
    `target gene ${r.gene} · ${r.cellType}`,
    `mean beta ${dec(r.mean)}`,
    `std ${dec(r.std)}`,
    `n = ${r.n == null ? '—' : r.n.toLocaleString()} cells`,
  ].join('\n');
}

/** A directed transporter pair, as one heatmap row. */
export interface BetaDirection {
  /** `${env}__${cell}` — unique within a cell-type block. */
  id: string;
  env: string;
  cell: string;
  /** Coefficient per target gene; missing genes are absent. */
  byGene: Record<string, BetaRow>;
  /** Largest |beta| across this row's genes — the row sort key. */
  peak: number;
}

/** One cell type's block: its directed pairs, plus the cell count behind them. */
export interface BetaBlock {
  cellType: string;
  nCells: number | null;
  directions: BetaDirection[];
}

/**
 * Group raw rows into per-cell-type blocks of directed pairs.
 * `pairKeys` selects which pairs to include (one for a gene-pair entity, many for a metabolite).
 */
export function groupBeta(
  byPair: Record<string, BetaRow[]>,
  pairKeys: string[],
  cellTypes: string[],
): BetaBlock[] {
  const rows = pairKeys.flatMap((k) => byPair[k] ?? []);
  const blocks: BetaBlock[] = [];

  for (const cellType of cellTypes) {
    const mine = rows.filter((r) => r.cellType === cellType);
    if (!mine.length) continue;

    const byDir = new Map<string, BetaDirection>();
    for (const r of mine) {
      const id = `${r.env}__${r.cell}`;
      let d = byDir.get(id);
      if (!d) {
        d = { id, env: r.env, cell: r.cell, byGene: {}, peak: 0 };
        byDir.set(id, d);
      }
      d.byGene[r.gene] = r;
      d.peak = Math.max(d.peak, Math.abs(r.mean ?? 0));
    }

    blocks.push({
      cellType,
      // n is a property of the cell type at this tier, so every row of the block agrees.
      nCells: mine[0].n,
      directions: [...byDir.values()].sort((a, b) => b.peak - a.peak),
    });
  }

  return blocks;
}
