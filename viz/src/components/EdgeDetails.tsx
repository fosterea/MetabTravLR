/** Canvas overlay: details for the clicked edge (an undirected cell-type interface) — its
 *  strength, significance, and cell types. Unit 3 will extend this with the metabolite→
 *  gene-pair breakdown at this interface. Cleared by clicking empty canvas or the close ×. */
import { useMemo } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { sameInterface, isSelfEdge } from '@/data/scales';
import { formatStrength, formatFdr } from '@/data/format';
import styles from './EdgeDetails.module.css';

export default function EdgeDetails() {
  const tier = useVizStore(selectCurrentTier);
  const entityId = useVizStore((s) => s.entityId);
  const entityKind = useVizStore((s) => s.entityKind);
  const edgeBundle = useVizStore((s) => s.edgeBundle);
  const selectedEdge = useVizStore((s) => s.selectedEdge);
  const selectEdge = useVizStore((s) => s.selectEdge);

  const edge = useMemo(() => {
    if (!selectedEdge || !entityId || !edgeBundle) return undefined;
    const list = edgeBundle.byEntity[entityId] ?? [];
    return list.find((e) => sameInterface(e, selectedEdge));
  }, [selectedEdge, entityId, edgeBundle]);

  if (!edge) return null;
  const self = isSelfEdge(edge);
  const s = edge.scores;

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
        {entityKind === 'metabolite' ? 'metabolite' : 'gene-pair'} interface
        {self ? ' · within cell type' : ''} · {tier?.label}
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
      <div className={styles.note}>Interfaces are undirected — order is not a direction.</div>
    </div>
  );
}
