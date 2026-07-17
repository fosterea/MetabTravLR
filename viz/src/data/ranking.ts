/** Entity list ordering for the side panel. Metabolites are ranked to surface the ones most
 *  involved at the current tier (serves the "which metabolites influence T cells?" question
 *  without hard-coding T cells — it uses the tier's own T-cell-involvement summary field). */
import type { Entity, GenePairEntity, MetaboliteEntity } from './types';

export interface RankedEntity {
  entity: Entity;
  /** Compact secondary label shown under the name. */
  hint: string;
  /** True if this entity is significant/involved at the current tier (for a badge). */
  flagged: boolean;
}

export function rankMetabolites(list: MetaboliteEntity[], tierId: string): RankedEntity[] {
  const scored = list.map((m) => {
    const t = m.perTier[tierId];
    const involved = t?.tcellInvolved === true;
    const nSig = t?.nSigPairs ?? 0;
    const rank =
      (involved ? 1_000_000 : 0) +
      (nSig ?? 0) * 1000 +
      (m.globalSignificant ? 100 : 0) +
      (m.nSigGenePairsGlobal ?? 0);
    const hint = t?.tcellInvolved
      ? `T-cell involved · ${nSig} sig pair${nSig === 1 ? '' : 's'}`
      : m.globalSignificant
        ? `global sig · ${m.nGenePairs} pairs`
        : `${m.nGenePairs} pairs`;
    return { entity: m as Entity, hint, flagged: involved || nSig > 0, rank };
  });
  scored.sort((a, b) => b.rank - a.rank || a.entity.id.localeCompare(b.entity.id));
  return scored.map(({ entity, hint, flagged }) => ({ entity, hint, flagged }));
}

export function rankGenePairs(list: GenePairEntity[]): RankedEntity[] {
  return [...list]
    .sort((a, b) => b.metabolites.length - a.metabolites.length || a.id.localeCompare(b.id))
    .map((g) => ({
      entity: g as Entity,
      hint: `${g.metabolites.length} metabolite${g.metabolites.length === 1 ? '' : 's'}`,
      flagged: g.metabolites.length > 0,
    }));
}

export const entityLabel = (e: Entity): string =>
  e.kind === 'metabolite' ? e.name : e.genes.join(' – ');

export function filterEntities(ranked: RankedEntity[], query: string): RankedEntity[] {
  const q = query.trim().toLowerCase();
  if (!q) return ranked;
  return ranked.filter((r) => entityLabel(r.entity).toLowerCase().includes(q));
}
