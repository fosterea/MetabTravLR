/** Entity list ordering for the side panel. Metabolites are ranked to surface the ones most
 *  involved at the current tier (serves the "which metabolites influence T cells?" question
 *  without hard-coding T cells — it uses the tier's own T-cell-involvement summary field). */
import type { Entity, EntityEdge, GenePairEntity, MetaboliteEntity } from './types';

/** entityId -> its edges at the current (tier, kind), i.e. `EdgeBundle.byEntity`. */
type ByEntity = Record<string, EntityEdge[]> | undefined;

const plural = (n: number) => (n === 1 ? '' : 's');

export interface RankedEntity {
  entity: Entity;
  /** Compact secondary label shown under the name. */
  hint: string;
  /** True if this entity is significant/involved at the current tier (for a badge). */
  flagged: boolean;
  /** True if the metabolite has NO significant interaction at ANY tier (nor globally): it is
   *  supported in the network but was eliminated for insignificance. Shown greyed, at the
   *  bottom, so the full network support stays visible. */
  eliminated?: boolean;
}

/** Significant somewhere — globally or at any tier. Independent of the currently-selected tier
 *  (a metabolite significant at another tier is NOT "eliminated", just not involved here). */
function metaboliteInvolvedAnywhere(m: MetaboliteEntity): boolean {
  if (m.globalSignificant) return true;
  if ((m.nSigGenePairsGlobal ?? 0) > 0) return true;
  for (const t of Object.values(m.perTier)) {
    if (t?.tcellInvolved) return true;
    if ((t?.nSigPairs ?? 0) > 0) return true;
  }
  return false;
}

export function rankMetabolites(
  list: MetaboliteEntity[],
  tierId: string,
  byEntity?: ByEntity,
): RankedEntity[] {
  const scored = list.map((m) => {
    const t = m.perTier[tierId];
    const involved = t?.tcellInvolved === true;
    const eliminated = !metaboliteInvolvedAnywhere(m);
    // # significant cell-type interfaces for this metabolite at the current tier (from the
    // loaded bundle); falls back to the summary sig-pair count while the bundle is loading.
    const nSigInt = byEntity ? (byEntity[m.id] ?? []).filter((e) => e.scores.selected).length : null;
    const nForRank = nSigInt ?? t?.nSigPairs ?? 0;
    const rank =
      (involved ? 1_000_000 : 0) +
      nForRank * 1000 +
      (m.globalSignificant ? 100 : 0) +
      (m.nSigGenePairsGlobal ?? 0);
    const hint = eliminated
      ? 'eliminated — not significant'
      : nSigInt != null
        ? `${nSigInt} sig interface${plural(nSigInt)}${involved ? ' · T-cell' : ''}`
        : involved
          ? 'T-cell involved'
          : m.globalSignificant
            ? 'global sig'
            : `${m.nGenePairs} pairs`;
    return {
      entity: m as Entity,
      hint,
      flagged: !eliminated && (involved || (nSigInt ?? 0) > 0),
      rank,
      eliminated,
    };
  });
  // Eliminated metabolites always sort below the significant ones (network support preserved).
  scored.sort(
    (a, b) =>
      Number(a.eliminated) - Number(b.eliminated) ||
      b.rank - a.rank ||
      a.entity.id.localeCompare(b.entity.id),
  );
  return scored.map(({ entity, hint, flagged, eliminated }) => ({
    entity,
    hint,
    flagged,
    eliminated,
  }));
}

export function rankGenePairs(list: GenePairEntity[], byEntity?: ByEntity): RankedEntity[] {
  const scored = list.map((g) => {
    // The gene_pair bundle is significant-only, so its edge count IS the number of significant
    // interactions at this tier. Pairs absent from the bundle have none (greyed, like eliminated
    // metabolites) — but all network pairs stay listed so the full support is visible.
    const nSigInt = byEntity ? (byEntity[g.id]?.length ?? 0) : null;
    const nMetab = g.metabolites.length;
    const eliminated = nSigInt === 0;
    const hint =
      nSigInt == null
        ? `${nMetab} metabolite${plural(nMetab)}`
        : nSigInt === 0
          ? `no sig interactions here · ${nMetab} metabolite${plural(nMetab)}`
          : `${nSigInt} sig interface${plural(nSigInt)} · ${nMetab} metabolite${plural(nMetab)}`;
    return { entity: g as Entity, hint, flagged: (nSigInt ?? 0) > 0, eliminated, nSigInt, nMetab };
  });
  // Sort by # significant interactions (desc), then metabolite count; greyed (0) sink to the
  // bottom. While the bundle is loading (nSigInt null) fall back to metabolite-count order.
  scored.sort(
    (a, b) =>
      Number(a.eliminated) - Number(b.eliminated) ||
      (b.nSigInt ?? 0) - (a.nSigInt ?? 0) ||
      b.nMetab - a.nMetab ||
      a.entity.id.localeCompare(b.entity.id),
  );
  return scored.map(({ entity, hint, flagged, eliminated }) => ({
    entity,
    hint,
    flagged,
    eliminated,
  }));
}

export const entityLabel = (e: Entity): string =>
  e.kind === 'metabolite' ? e.name : e.genes.join(' – ');

export function filterEntities(ranked: RankedEntity[], query: string): RankedEntity[] {
  const q = query.trim().toLowerCase();
  if (!q) return ranked;
  return ranked.filter((r) => entityLabel(r.entity).toLowerCase().includes(q));
}
