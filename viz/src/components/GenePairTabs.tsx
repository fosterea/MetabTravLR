/** Tab strip over the canvas: for the selected metabolite, one tab per transporter gene pair
 *  that is significant at the current tier, plus an "All" tab (the metabolite itself). Picking
 *  a pair tab isolates that single pair's interfaces on the graph — a per-pair drill-down that
 *  complements the on-graph fan-out and the panel list. Hidden for gene-pair view / no metab. */
import { useMemo } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { formatStrength } from '@/data/format';
import styles from './GenePairTabs.module.css';

export default function GenePairTabs() {
  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const gpBundle = useVizStore((s) => s.gpBundle);
  const gpTab = useVizStore((s) => s.gpTab);
  const setGpTab = useVizStore((s) => s.setGpTab);

  const metab = useMemo(
    () =>
      entityKind === 'metabolite'
        ? dataset?.entities.metabolite?.find((m) => m.id === entityId)
        : undefined,
    [dataset, entityKind, entityId],
  );

  // This metabolite's gene pairs that have any significant interface at the current tier.
  const pairs = useMemo(() => {
    if (!metab || !gpBundle || !tier) return [];
    const present = new Set(tier.cellTypes);
    const seen = new Set<string>();
    const out: { id: string; label: string; nInterfaces: number; maxC: number }[] = [];
    for (const [a, b] of metab.genePairs) {
      const canon = a <= b ? `${a}__${b}` : `${b}__${a}`;
      if (seen.has(canon)) continue;
      seen.add(canon);
      const primary = `${a}__${b}`;
      const useId = gpBundle.byEntity[primary] ? primary : `${b}__${a}`;
      const edges = (gpBundle.byEntity[useId] ?? []).filter(
        (e) => present.has(e.source) && present.has(e.target),
      );
      if (!edges.length) continue;
      out.push({
        id: useId,
        label: `${a} – ${b}`,
        nInterfaces: edges.length,
        maxC: edges.reduce((m, e) => Math.max(m, e.scores.C_np), 0),
      });
    }
    out.sort((x, y) => y.maxC - x.maxC);
    return out;
  }, [metab, gpBundle, tier]);

  if (!metab || pairs.length === 0) return null;

  return (
    <div className={styles.bar} role="tablist" aria-label="Gene pairs of this metabolite">
      <span className={styles.label}>Gene pairs:</span>
      <button
        role="tab"
        aria-selected={!gpTab}
        className={`${styles.tab} ${!gpTab ? styles.active : ''}`}
        onClick={() => setGpTab(undefined)}
      >
        All <span className={styles.count}>metabolite</span>
      </button>
      {pairs.map((p) => (
        <button
          key={p.id}
          role="tab"
          aria-selected={gpTab === p.id}
          className={`${styles.tab} ${gpTab === p.id ? styles.active : ''}`}
          onClick={() => setGpTab(p.id)}
          title={`${p.nInterfaces} interface${p.nInterfaces === 1 ? '' : 's'} · max C_np ${formatStrength(p.maxC)}`}
        >
          {p.label} <span className={styles.count}>{p.nInterfaces}</span>
        </button>
      ))}
    </div>
  );
}
