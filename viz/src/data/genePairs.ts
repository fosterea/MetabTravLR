/** Metabolite → transporter-gene-pair expansion, shared by the graph fan-out and the
 *  details-panel breakdown. A metabolite's per-interface communication is the sum of its
 *  gene pairs' scores (harreman `compute_metabolite_cs`); this recovers which of those pairs
 *  are individually significant at a given cell-type interface, from the same-tier gene_pair
 *  bundle (which is significant-only, see docs/05_data_contract.md). No source/ingest change. */
import type { EdgeBundle, EntityEdge, MetaboliteEntity, Tier } from './types';
import { sameInterface } from './scales';

export interface GpContribution {
  id: string; // "GENE1__GENE2"
  genes: [string, string];
  label: string; // "GENE1 – GENE2"
  edge: EntityEdge; // the gene-pair edge at this interface (carries its own scores)
}

/** Reverse a "A__B" gene-pair id (order-tolerant bundle lookups). */
export const reverseGpId = (id: string): string => {
  const i = id.indexOf('__');
  return i < 0 ? id : `${id.slice(i + 2)}__${id.slice(0, i)}`;
};

/** Edges of a gene pair (id order-tolerant) whose endpoints are both in the tier. */
export function gpEdgesInTier(
  gpBundle: EdgeBundle | undefined,
  id: string,
  cellTypes: string[],
): EntityEdge[] {
  if (!gpBundle) return [];
  const present = new Set(cellTypes);
  const rows = gpBundle.byEntity[id] ?? gpBundle.byEntity[reverseGpId(id)] ?? [];
  return rows.filter((e) => present.has(e.source) && present.has(e.target));
}

export interface MetaboliteGpAtTier {
  id: string; // "GENE1__GENE2" (network order)
  label: string; // "GENE1 – GENE2"
  genes: [string, string];
  nInterfaces: number;
  maxC: number;
}

/**
 * The selected metabolite's transporter gene pairs that are significant at the given tier,
 * ordered by strength (maxC desc). Shared by the gene-pair tabs and the panel breakdown.
 */
export function metaboliteSigPairsAtTier(
  metab: MetaboliteEntity | undefined,
  gpBundle: EdgeBundle | undefined,
  tier: Tier | undefined,
): MetaboliteGpAtTier[] {
  if (!metab || !gpBundle || !tier) return [];
  const seen = new Set<string>();
  const out: MetaboliteGpAtTier[] = [];
  for (const [a, b] of metab.genePairs) {
    const canon = a <= b ? `${a}__${b}` : `${b}__${a}`;
    if (seen.has(canon)) continue;
    seen.add(canon);
    const id = `${a}__${b}`;
    const edges = gpEdgesInTier(gpBundle, id, tier.cellTypes);
    if (!edges.length) continue;
    out.push({
      id,
      label: `${a} – ${b}`,
      genes: [a, b],
      nInterfaces: edges.length,
      maxC: edges.reduce((m, e) => Math.max(m, e.scores.C_np), 0),
    });
  }
  out.sort((x, y) => y.maxC - x.maxC);
  return out;
}

/**
 * Gene pairs of `metab` that have a significant edge at `iface` in the same-tier gene_pair
 * bundle, sorted by strength (C_np) desc. Returns [] if the bundle isn't loaded or none
 * qualify (a metabolite edge can be significant as a SUM while no single pair is).
 */
export function genePairsAtInterface(
  metab: MetaboliteEntity,
  gpBundle: EdgeBundle | undefined,
  iface: { source: string; target: string },
): GpContribution[] {
  if (!gpBundle) return [];
  const out: GpContribution[] = [];
  const seen = new Set<string>();
  for (const [a, b] of metab.genePairs) {
    const id = `${a}__${b}`;
    // Dedup on the CANONICAL (order-independent) key so a pair listed in both gene orders
    // can't be counted twice (both orders resolve to the same undirected gp edge).
    const key = a <= b ? `${a}__${b}` : `${b}__${a}`;
    if (seen.has(key)) continue;
    seen.add(key);
    // The pair id order in the network vs the tier CSV should agree, but check both to be safe.
    const edges = gpBundle.byEntity[id] ?? gpBundle.byEntity[`${b}__${a}`] ?? [];
    const at = edges.find((e) => sameInterface(e, iface));
    if (at) out.push({ id, genes: [a, b], label: `${a} – ${b}`, edge: at });
  }
  out.sort((x, y) => y.edge.scores.C_np - x.edge.scores.C_np);
  return out;
}
