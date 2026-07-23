/** Left sidebar: searchable, ranked list of entities (metabolites / gene pairs). Selecting an
 *  item drives the graph. Metabolites are ranked to surface tier-relevant ones first. */
import { Fragment, useMemo, useState } from 'react';
import { useVizStore } from '@/store/useVizStore';
import {
  entityLabel,
  filterEntities,
  rankGenePairs,
  rankMetabolites,
  type RankedEntity,
} from '@/data/ranking';
import styles from './EntityPanel.module.css';

export default function EntityPanel() {
  const dataset = useVizStore((s) => s.dataset);
  const tierId = useVizStore((s) => s.tierId);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const edgeBundle = useVizStore((s) => s.edgeBundle);
  const fdrThreshold = useVizStore((s) => s.fdrThreshold);
  const selectEntity = useVizStore((s) => s.selectEntity);
  const [query, setQuery] = useState('');

  // byEntity is the current (tier, kind) bundle, so counts/sorting update per tier.
  const byEntity = edgeBundle?.byEntity;
  const ranked: RankedEntity[] = useMemo(() => {
    if (!dataset) return [];
    if (entityKind === 'metabolite') {
      return rankMetabolites(dataset.entities.metabolite ?? [], tierId ?? '', byEntity, fdrThreshold);
    }
    return rankGenePairs(dataset.entities.gene_pair ?? [], byEntity, fdrThreshold);
  }, [dataset, tierId, entityKind, byEntity, fdrThreshold]);

  const dividerLabel =
    entityKind === 'metabolite'
      ? 'Eliminated — not significant (full network support)'
      : 'No significant interactions at this tier (full support)';

  const visible = useMemo(() => filterEntities(ranked, query), [ranked, query]);

  return (
    <aside className="sidebar">
      <div className={styles.head}>
        <div className={styles.title}>
          {entityKind === 'metabolite' ? 'Metabolites' : 'Gene pairs'}
          <span className="muted"> · {visible.length}</span>
        </div>
        <input
          className="control"
          type="search"
          placeholder={`Search ${entityKind === 'metabolite' ? 'metabolites' : 'gene pairs'}…`}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search entities"
        />
        {entityKind === 'metabolite' && (
          <div className={styles.subtle}>Ranked by involvement at the selected tier</div>
        )}
      </div>

      <ul className={styles.list}>
        {visible.map((r, i) => {
          const id = r.entity.id;
          const selected = id === entityId;
          // Divider announcing the greyed, insignificant tail (full network support shown).
          const startEliminated = r.eliminated && !visible[i - 1]?.eliminated;
          return (
            <Fragment key={id}>
              {startEliminated && (
                <li className={styles.divider} aria-hidden>
                  {dividerLabel}
                </li>
              )}
              <li>
                <button
                  className={`${styles.item} ${selected ? styles.selected : ''} ${
                    r.eliminated ? styles.eliminated : ''
                  }`}
                  aria-pressed={selected}
                  onClick={() => selectEntity(id)}
                >
                  <span className={styles.name}>{entityLabel(r.entity)}</span>
                  <span className={styles.hint}>
                    {r.flagged && <span className={styles.dot} aria-hidden />}
                    {r.hint}
                  </span>
                </button>
              </li>
            </Fragment>
          );
        })}
        {visible.length === 0 && <li className={styles.subtle}>No matches.</li>}
      </ul>
    </aside>
  );
}
