/** Tab strip over the canvas: for the selected metabolite, one tab per transporter gene pair
 *  that is significant at the current tier, plus an "All" tab (the metabolite itself). Picking
 *  a pair tab isolates that single pair's interfaces on the graph — a per-pair drill-down that
 *  complements the on-graph fan-out and the panel list. Hidden for gene-pair view / no metab. */
import { useEffect, useMemo, useRef } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { metaboliteSigPairsAtTier, type MetaboliteGpAtTier } from '@/data/genePairs';
import { formatStrength } from '@/data/format';
import styles from './GenePairTabs.module.css';

export default function GenePairTabs() {
  const activeRef = useRef<HTMLButtonElement>(null);
  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const gpBundle = useVizStore((s) => s.gpBundle);
  const gpTab = useVizStore((s) => s.gpTab);
  const setGpTab = useVizStore((s) => s.setGpTab);
  const fdrThreshold = useVizStore((s) => s.fdrThreshold);

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
    () => metaboliteSigPairsAtTier(metab, gpBundle, tier, fdrThreshold),
    [metab, gpBundle, tier, fdrThreshold],
  );

  // Which tabs to render. Tightening the FDR cutoff can drop the isolated pair's last
  // significant interface, which removes it from `pairs` — but a pair the user explicitly
  // picked KEEPS its tab (flagged "not significant") instead of vanishing. Only an explicit
  // tab switch, or a metabolite/tier/dataset/kind change (all of which clear `gpTab` in the
  // store), may take it away. Silently falling back to "All" lost the user's place mid-drag
  // and read as the graph forwarding itself back to the metabolite view.
  const tabs = useMemo<MetaboliteGpAtTier[]>(() => {
    if (!gpTab || !metab || pairs.some((p) => p.id === gpTab)) return pairs;
    // Belt-and-braces: only retain a tab that really is one of THIS metabolite's pairs.
    const genes = metab.genePairs.find(
      ([a, b]) => `${a}__${b}` === gpTab || `${b}__${a}` === gpTab,
    );
    if (!genes) return pairs;
    // `metaboliteSigPairsAtTier` derives id and label from the same [a,b]; order the genes the
    // way `gpTab` spells them so the retained tab reads exactly as it did before the cutoff
    // emptied it.
    const [a, b] = genes;
    const ordered: [string, string] = `${a}__${b}` === gpTab ? [a, b] : [b, a];
    return [
      ...pairs,
      {
        id: gpTab,
        label: `${ordered[0]} – ${ordered[1]}`,
        genes: ordered,
        nInterfaces: 0,
        maxC: 0,
      },
    ];
  }, [pairs, gpTab, metab]);

  // Keep the active tab in view: the strip scrolls horizontally, and a retained (emptied) tab
  // sorts last, so it could otherwise sit off-screen — leaving no visible tab looking selected,
  // which is the very confusion this change exists to prevent.
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [gpTab, tabs]);

  if (!metab || tabs.length === 0) return null;

  return (
    <div className={styles.bar} role="tablist" aria-label="Gene pairs of this metabolite">
      <span className={styles.label}>Gene pairs:</span>
      <button
        ref={!gpTab ? activeRef : undefined}
        role="tab"
        aria-selected={!gpTab}
        className={`${styles.tab} ${!gpTab ? styles.active : ''}`}
        onClick={() => setGpTab(undefined)}
      >
        All <span className={styles.count}>metabolite</span>
      </button>
      {tabs.map((p) => {
        // nInterfaces 0 ⇒ this is the retained tab: still selected, but nothing passes the
        // current cutoff. Say so in the tab rather than dropping it.
        const emptied = p.nInterfaces === 0;
        return (
          <button
            key={p.id}
            ref={gpTab === p.id ? activeRef : undefined}
            role="tab"
            aria-selected={gpTab === p.id}
            className={`${styles.tab} ${gpTab === p.id ? styles.active : ''} ${emptied ? styles.emptied : ''}`}
            onClick={() => setGpTab(p.id)}
            title={
              emptied
                ? `No interface significant at FDR < ${fdrThreshold} — loosen the cutoff, or pick another tab`
                : `${p.nInterfaces} interface${p.nInterfaces === 1 ? '' : 's'} · max C_np ${formatStrength(p.maxC)}`
            }
          >
            {p.label}{' '}
            <span className={styles.count}>{emptied ? 'not significant' : p.nInterfaces}</span>
          </button>
        );
      })}
    </div>
  );
}
