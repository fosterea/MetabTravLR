/** Metabolite → transporter-gene-pair expansion, shared by the graph fan-out and the
 *  details-panel breakdown. A metabolite's per-interface communication is the sum of its
 *  gene pairs' scores (harreman `compute_metabolite_cs`); this recovers which of those pairs
 *  are individually significant at a given cell-type interface, from the same-tier gene_pair
 *  bundle (which is significant-only, see docs/05_data_contract.md). No source/ingest change. */
import type { EdgeBundle, EntityEdge, MetaboliteEntity } from './types';
import { sameInterface } from './scales';

export interface GpContribution {
  id: string; // "GENE1__GENE2"
  genes: [string, string];
  label: string; // "GENE1 – GENE2"
  edge: EntityEdge; // the gene-pair edge at this interface (carries its own scores)
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
    if (seen.has(id)) continue;
    seen.add(id);
    // The pair id order in the network vs the tier CSV should agree, but check both to be safe.
    const edges = gpBundle.byEntity[id] ?? gpBundle.byEntity[`${b}__${a}`] ?? [];
    const at = edges.find((e) => sameInterface(e, iface));
    if (at) out.push({ id, genes: [a, b], label: `${a} – ${b}`, edge: at });
  }
  out.sort((x, y) => y.edge.scores.C_np - x.edge.scores.C_np);
  return out;
}
