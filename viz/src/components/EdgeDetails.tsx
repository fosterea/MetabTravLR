/** Canvas overlay: details for the clicked edge (an undirected cell-type interface) — its
 *  strength, significance, cell types, and the transporter gene pairs that carry it at this
 *  interface. The pair list shows in BOTH gene-pair expand modes: "On graph" additionally fans
 *  the pairs out as sub-edges, and the two stay linked (hover/click one, the other highlights).
 *  Each pair is a link to its own gene-pair view. Cleared by clicking empty canvas or the × . */
import { useMemo } from 'react';
import { useVizStore, selectCurrentTier, selectFocusedGp } from '@/store/useVizStore';
import { sameInterface, isSelfEdge, isSelected } from '@/data/scales';
import { genePairsAtInterface, gpEdgesInTier } from '@/data/genePairs';
import { formatStrength, formatFdr } from '@/data/format';
import styles from './EdgeDetails.module.css';

export default function EdgeDetails() {
  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityId = useVizStore((s) => s.entityId);
  const entityKind = useVizStore((s) => s.entityKind);
  const edgeBundle = useVizStore((s) => s.edgeBundle);
  const gpBundle = useVizStore((s) => s.gpBundle);
  const showNonSignificant = useVizStore((s) => s.showNonSignificant);
  const fdrThreshold = useVizStore((s) => s.fdrThreshold);
  const gpExpandMode = useVizStore((s) => s.gpExpandMode);
  const gpTab = useVizStore((s) => s.gpTab);
  const selectedEdge = useVizStore((s) => s.selectedEdge);
  const selectEdge = useVizStore((s) => s.selectEdge);
  const focusedGp = useVizStore(selectFocusedGp);
  const pinnedGp = useVizStore((s) => s.pinnedGp);
  const setHoverGp = useVizStore((s) => s.setHoverGp);
  const goToEntity = useVizStore((s) => s.goToEntity);

  // Resolve against the SAME visible set the graph renders, so the panel and the canvas never
  // disagree (e.g. after toggling non-significant off, a now-hidden pick shows no panel). When
  // a gene-pair tab is isolated, resolve against that pair's edges instead.
  const edge = useMemo(() => {
    if (!selectedEdge || !tier) return undefined;
    if (gpTab && gpBundle) {
      return gpEdgesInTier(gpBundle, gpTab, tier.cellTypes).find((e) =>
        sameInterface(e, selectedEdge),
      );
    }
    if (!entityId || !edgeBundle) return undefined;
    const present = new Set(tier.cellTypes);
    const visible = (edgeBundle.byEntity[entityId] ?? []).filter(
      (e) =>
        present.has(e.source) &&
        present.has(e.target) &&
        (showNonSignificant || isSelected(e.scores, fdrThreshold)),
    );
    return visible.find((e) => sameInterface(e, selectedEdge));
  }, [selectedEdge, entityId, edgeBundle, gpTab, gpBundle, tier, showNonSignificant, fdrThreshold]);

  const gpTabLabel = useMemo(() => {
    if (!gpTab) return undefined;
    const gp = dataset?.entities.gene_pair?.find((g) => g.id === gpTab);
    return gp ? gp.genes.join(' – ') : gpTab.replace('__', ' – ');
  }, [gpTab, dataset]);

  const metab = useMemo(
    () =>
      entityKind === 'metabolite'
        ? dataset?.entities.metabolite?.find((m) => m.id === entityId)
        : undefined,
    [dataset, entityKind, entityId],
  );

  // Gene-pair breakdown for a metabolite interface. Shown in BOTH expand modes — "On graph"
  // adds the fan-out, it doesn't replace the list (the list is where the numbers and the links
  // live). Empty in the gene-pair view, or when a single pair tab is already isolated.
  const gps = useMemo(
    () => (edge && metab && !gpTab ? genePairsAtInterface(metab, gpBundle, edge) : []),
    [edge, metab, gpBundle, gpTab],
  );

  if (!edge) return null;
  const self = isSelfEdge(edge);
  const s = edge.scores;
  const showBreakdown = !!metab && !gpTab;

  return (
    <div className={styles.panel} aria-label="Selected interface details">
      <div className={styles.head}>
        <div className={styles.title}>
          {self ? (
            <>
              within <b>{edge.source}</b>
            </>
          ) : (
            <>
              <b>{edge.source}</b> ↔ <b>{edge.target}</b>
            </>
          )}
        </div>
        <button className={styles.close} onClick={() => selectEdge(undefined)} aria-label="Close">
          ×
        </button>
      </div>
      <div className={styles.sub}>
        {gpTab
          ? `gene pair ${gpTabLabel}`
          : entityKind === 'metabolite'
            ? 'metabolite'
            : 'gene-pair'}{' '}
        interface{self ? ' · within cell type' : ''} · {tier?.label}
      </div>
      <dl className={styles.rows}>
        <div className={styles.row}>
          <dt>Strength</dt>
          <dd>
            <span className={styles.strong}>{formatStrength(s.C_np)}</span>{' '}
            <span className="muted">C_np</span>
          </dd>
        </div>
        <div className={styles.row}>
          <dt>Significance</dt>
          <dd>
            {isSelected(s, fdrThreshold) ? (
              <span className={styles.sig}>significant</span>
            ) : (
              <span className="muted">not significant</span>
            )}{' '}
            <span className="muted">· FDR {formatFdr(s.FDR_np)}</span>
          </dd>
        </div>
        <div className={styles.row}>
          <dt>Parametric</dt>
          <dd className="muted">
            C_p {formatStrength(s.C_p)} · Z {formatStrength(s.Z)}
          </dd>
        </div>
      </dl>

      {showBreakdown && (
        <div className={styles.gp}>
          {gps.length > 0 ? (
            <>
              <div className={styles.gpHead}>
                Carried by {gps.length} significant transporter pair{gps.length === 1 ? '' : 's'}
              </div>
              <ul className={styles.gpList}>
                {gps.map((g) => (
                  <li key={g.id}>
                    <button
                      className={`${styles.gpItem} ${focusedGp === g.id ? styles.gpFocused : ''} ${
                        pinnedGp === g.id ? styles.gpPinned : ''
                      }`}
                      // Hovering previews the pair on the graph's fan-out; clicking opens the
                      // pair's own view (the per-kind memory makes coming back one click).
                      onMouseEnter={() => setHoverGp(g.id)}
                      onMouseLeave={() => setHoverGp(undefined)}
                      onFocus={() => setHoverGp(g.id)}
                      onBlur={() => setHoverGp(undefined)}
                      onClick={() => void goToEntity('gene_pair', g.id)}
                      title={`Open the ${g.label} gene-pair view`}
                    >
                      <span className={styles.gpName}>{g.label}</span>
                      <span className={styles.gpVal}>{formatStrength(g.edge.scores.C_np)}</span>
                    </button>
                  </li>
                ))}
              </ul>
              <div className={styles.gpFoot}>
                Significant subset — pair strengths need not sum to the metabolite total.
                {gpExpandMode === 'graph' && ' Hover a row to find it in the on-graph fan.'}
              </div>
            </>
          ) : (
            <div className="muted">No individually-significant transporter pairs at this interface.</div>
          )}
        </div>
      )}

      <div className={styles.note}>Interfaces are undirected — order is not a direction.</div>
    </div>
  );
}
