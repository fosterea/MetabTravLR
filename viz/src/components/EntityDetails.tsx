/** Canvas overlay: details for the selected entity + a summary of the edges in view. */
import { useMemo } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import styles from './EntityDetails.module.css';

export default function EntityDetails() {
  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const edgeBundle = useVizStore((s) => s.edgeBundle);

  const entity = useMemo(() => {
    if (!dataset || !entityId) return undefined;
    const list =
      entityKind === 'metabolite' ? dataset.entities.metabolite : dataset.entities.gene_pair;
    return list?.find((e) => e.id === entityId);
  }, [dataset, entityId, entityKind]);

  const edges = (entityId && edgeBundle?.byEntity[entityId]) || [];
  const nSig = edges.filter((e) => e.scores.selected).length;

  if (!entity) return null;

  return (
    <div className={styles.panel} aria-label="Selected entity details">
      {entity.kind === 'metabolite' ? (
        <>
          <div className={styles.name}>{entity.name}</div>
          <div className={styles.meta}>
            {entity.globalSignificant ? (
              <span className={styles.badge}>globally significant</span>
            ) : (
              <span className={`${styles.badge} ${styles.badgeMuted}`}>not global-sig</span>
            )}
            <span className="muted">{entity.nGenePairs} transporter pairs</span>
          </div>
          {tier && (
            <div className={styles.metric}>
              At {tier.label}:{' '}
              {entity.perTier[tier.id]?.tcellInvolved ? 'T-cell involved' : 'no T-cell involvement'}
              {entity.perTier[tier.id]?.interactions && (
                <div className={styles.small}>{entity.perTier[tier.id]?.interactions}</div>
              )}
            </div>
          )}
        </>
      ) : (
        <>
          <div className={styles.name}>{entity.genes.join(' – ')}</div>
          <div className={styles.meta}>
            <span className="muted">transporter gene pair</span>
          </div>
          <div className={styles.metric}>
            Serves {entity.metabolites.length} metabolite
            {entity.metabolites.length === 1 ? '' : 's'} (many-to-many):
            <div className={styles.chips}>
              {entity.metabolites.slice(0, 8).map((m) => (
                <span key={m} className={styles.chip}>
                  {m}
                </span>
              ))}
              {entity.metabolites.length > 8 && (
                <span className="muted">+{entity.metabolites.length - 8}</span>
              )}
            </div>
          </div>
        </>
      )}
      <div className={styles.summary}>
        <strong>{nSig}</strong> significant / {edges.length} interface
        {edges.length === 1 ? '' : 's'} at {tier?.label}
      </div>
    </div>
  );
}
