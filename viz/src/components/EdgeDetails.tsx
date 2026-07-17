/** Canvas overlay: details for the clicked edge (an undirected cell-type interface) — its
 *  strength, significance, cell types, and (metabolite + "In panel" mode) the transporter
 *  gene pairs that carry it at this interface. Cleared by clicking empty canvas or the × . */
import { useMemo } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { sameInterface, isSelfEdge } from '@/data/scales';
import { genePairsAtInterface } from '@/data/genePairs';
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
  const gpExpandMode = useVizStore((s) => s.gpExpandMode);
  const gpTab = useVizStore((s) => s.gpTab);
  const selectedEdge = useVizStore((s) => s.selectedEdge);
  const selectEdge = useVizStore((s) => s.selectEdge);

  // Resolve against the SAME visible set the graph renders, so the panel and the canvas never
  // disagree (e.g. after toggling non-significant off, a now-hidden pick shows no panel). When
  // a gene-pair tab is isolated, resolve against that pair's edges instead.
  const edge = useMemo(() => {
    if (!selectedEdge || !tier) return undefined;
    const present = new Set(tier.cellTypes);
    if (gpTab && gpBundle) {
      const list = (gpBundle.byEntity[gpTab] ?? []).filter(
        (e) => present.has(e.source) && present.has(e.target),
      );
      return list.find((e) => sameInterface(e, selectedEdge));
    }
    if (!entityId || !edgeBundle) return undefined;
    const visible = (edgeBundle.byEntity[entityId] ?? []).filter(
      (e) =>
        present.has(e.source) &&
        present.has(e.target) &&
        (showNonSignificant || e.scores.selected),
    );
    return visible.find((e) => sameInterface(e, selectedEdge));
  }, [selectedEdge, entityId, edgeBundle, gpTab, gpBundle, tier, showNonSignificant]);

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

  // Panel-mode gene-pair breakdown for a metabolite interface (empty in graph mode, gp view,
  // or when a single gene-pair tab is already isolated).
  const gps = useMemo(
    () =>
      edge && metab && gpExpandMode === 'panel' && !gpTab
        ? genePairsAtInterface(metab, gpBundle, edge)
        : [],
    [edge, metab, gpBundle, gpExpandMode, gpTab],
  );

  if (!edge) return null;
  const self = isSelfEdge(edge);
  const s = edge.scores;
  const showBreakdown = !!metab && gpExpandMode === 'panel' && !gpTab;

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
            {s.selected ? (
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
                  <li key={g.id} className={styles.gpItem}>
                    <span className={styles.gpName}>{g.label}</span>
                    <span className={styles.gpVal}>{formatStrength(g.edge.scores.C_np)}</span>
                  </li>
                ))}
              </ul>
              <div className={styles.gpFoot}>
                Significant subset — pair strengths need not sum to the metabolite total.
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
