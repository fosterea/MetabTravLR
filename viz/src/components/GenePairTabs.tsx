/** Tab strip over the canvas: for the selected metabolite, one tab per transporter gene pair
 *  that is significant at the current tier, plus an "All" tab (the metabolite itself). Picking
 *  a pair tab isolates that single pair's interfaces on the graph — a per-pair drill-down that
 *  complements the on-graph fan-out and the panel list. Hidden for gene-pair view / no metab. */
import { useMemo } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { metaboliteSigPairsAtTier } from '@/data/genePairs';
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

  // This metabolite's gene pairs significant at the current tier, with their stable color slot
  // (shared with the on-graph fan-out so a pair's color matches its tab swatch).
  const pairs = useMemo(
    () => metaboliteSigPairsAtTier(metab, gpBundle, tier),
    [metab, gpBundle, tier],
  );

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
          <span
            className={styles.swatch}
            style={{ background: `var(--gp-${p.slot + 1})` }}
            aria-hidden
          />
          {p.label} <span className={styles.count}>{p.nInterfaces}</span>
        </button>
      ))}
    </div>
  );
}
