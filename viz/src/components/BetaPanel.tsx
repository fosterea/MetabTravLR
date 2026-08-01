/**
 * SpaceTravLR coefficients for the selected entity, as a cell-type-major heatmap (`BetaMatrix`).
 *
 * The primary factor group is the metabolite's (or gene pair's) own transporter pairs, sourced from
 * the bundle's `metab` channel. Beneath it, the user can add COMPARISON factor groups — the other
 * feature channels (ligand–receptor, ligand–TF, transcription factors) and the full "all metabolic
 * transporters" superset — as chips. All groups are stacked cell-type-major by `BetaMatrix`, sharing
 * one UNION set of target-gene columns (so a gene only a comparison factor covers still appears),
 * while each factor keeps its OWN magnitude scale. Comparisons reset when the entity/tier/dataset
 * changes, but survive a cell-type-filter change.
 *
 * Both directions of a pair are separate rows on purpose: `A→B` and `B→A` are different
 * coefficients, and merging them would invent a symmetry the model does not claim.
 */
import { useEffect, useMemo, useState } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { betaKey } from '@/data/betaScale';
import type { BetaChannel, BetaChannelId, Entity } from '@/data/types';
import BetaMatrix, { type BetaFactorGroup } from './BetaMatrix';
import styles from './BetaPanel.module.css';

const ALL = '__all__';

/** The pair keys whose coefficients belong to this entity. */
function pairKeysFor(entity: Entity): string[] {
  if (entity.kind === 'gene_pair') return [betaKey(entity.genes[0], entity.genes[1])];
  // A metabolite is served by many transporter pairs (many-to-many); show all of them.
  return [...new Set(entity.genePairs.map(([a, b]) => betaKey(a, b)))];
}

/**
 * Chip/menu label for a channel. The primary section already shows THIS metabolite's own
 * transporter pairs, so the metab comparison shows *every* pair — hence a distinct label.
 */
function optionLabel(ch: BetaChannel): string {
  return ch.id === 'metab' ? 'All metabolic transporters' : ch.label;
}

export default function BetaPanel() {
  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const betaBundle = useVizStore((s) => s.betaBundle);
  const [cellTypeFilter, setCellTypeFilter] = useState<string>(ALL);

  // Comparison channels stacked below the primary, in the order they were added.
  const [added, setAdded] = useState<BetaChannelId[]>([]);
  // Stale comparisons must not linger onto a different entity/tier/dataset (their genes/scales are
  // no longer meaningful). A cell-type-filter change deliberately does NOT reset them.
  const datasetId = dataset?.id;
  const tierId = tier?.id;
  useEffect(() => {
    setAdded([]);
  }, [entityId, tierId, datasetId]);

  const entity = useMemo<Entity | undefined>(() => {
    if (!dataset || !entityId) return undefined;
    const list =
      entityKind === 'metabolite' ? dataset.entities.metabolite : dataset.entities.gene_pair;
    return list?.find((e) => e.id === entityId);
  }, [dataset, entityId, entityKind]);

  const metab = useMemo(() => betaBundle?.channels.find((c) => c.id === 'metab'), [betaBundle]);
  const channelsById = useMemo(() => {
    const m = new Map<BetaChannelId, BetaChannel>();
    for (const c of betaBundle?.channels ?? []) m.set(c.id, c);
    return m;
  }, [betaBundle]);

  // This entity's own transporter-pair rows, filtered out of the metab channel by sorted pair key.
  const primaryRows = useMemo(() => {
    if (!entity || !metab) return [];
    const keys = new Set(pairKeysFor(entity));
    return metab.rows.filter((r) => r.b != null && keys.has(betaKey(r.a, r.b)));
  }, [entity, metab]);

  // The stacked factor groups: the metabolite's own pairs first, then each added comparison. For
  // the "All metabolic transporters" option we compare against the WHOLE metab channel, not just
  // this metabolite's pairs.
  const groups = useMemo<BetaFactorGroup[]>(() => {
    if (!metab) return [];
    const primaryLabel = `This ${entityKind === 'metabolite' ? 'metabolite' : 'gene pair'}’s transporter pairs`;
    const primary: BetaFactorGroup = {
      key: 'metab-primary',
      channel: metab,
      label: primaryLabel,
      rows: primaryRows,
    };
    const rest = added
      .map((id): BetaFactorGroup | null => {
        const ch = channelsById.get(id);
        if (!ch) return null;
        return { key: id, channel: ch, label: optionLabel(ch), rows: id === 'metab' ? metab.rows : ch.rows };
      })
      .filter((g): g is BetaFactorGroup => g !== null);
    return [primary, ...rest];
  }, [metab, entityKind, primaryRows, added, channelsById]);

  // Every cell type that appears in ANY shown group, in the bundle's order.
  const availableCellTypes = useMemo(() => {
    if (!betaBundle) return [];
    const present = new Set<string>();
    for (const g of groups) for (const r of g.rows) present.add(r.cellType);
    return betaBundle.cellTypes.filter((ct) => present.has(ct));
  }, [betaBundle, groups]);

  // Cell-type names differ per tier, so a filter set at Tier3 is meaningless at Tier1. Fall back
  // to "All" rather than blanking the panel (and keep the <select> showing that fallback).
  const effectiveFilter = availableCellTypes.includes(cellTypeFilter) ? cellTypeFilter : ALL;
  const shown = useMemo(
    () => (effectiveFilter === ALL ? availableCellTypes : [effectiveFilter]),
    [effectiveFilter, availableCellTypes],
  );

  // Channels offered by the "Add comparison" menu: those in the bundle not already added.
  const availableToAdd = useMemo(
    () => (betaBundle?.channels ?? []).filter((c) => !added.includes(c.id)),
    [betaBundle, added],
  );

  if (!dataset?.hasBeta || !betaBundle) return null;
  if (!entity) return null;

  if (!metab || !availableCellTypes.length) {
    return (
      <section className={styles.wrap} aria-labelledby="beta-h">
        <h3 className={styles.title} id="beta-h">
          SpaceTravLR coefficients
        </h3>
        <div className={styles.none}>
          No SpaceTravLR coefficients for this{' '}
          {entityKind === 'metabolite' ? 'metabolite’s transporter pairs' : 'gene pair'} at{' '}
          {tier?.label}.
        </div>
      </section>
    );
  }

  const remove = (id: BetaChannelId) => setAdded((a) => a.filter((x) => x !== id));

  return (
    <section className={styles.wrap} aria-labelledby="beta-h">
      <div className={styles.head}>
        <div>
          <h3 className={styles.title} id="beta-h">
            SpaceTravLR coefficients
          </h3>
          <div className={styles.subtitle}>
            How much each transporter pair moves the target genes, per cell type
          </div>
        </div>

        <label className={styles.filter}>
          <span className={styles.filterLabel}>Cell type</span>
          <select
            className="control"
            value={effectiveFilter}
            onChange={(e) => setCellTypeFilter(e.target.value)}
          >
            <option value={ALL}>All ({availableCellTypes.length})</option>
            {availableCellTypes.map((ct) => (
              <option key={ct} value={ct}>
                {ct}
              </option>
            ))}
          </select>
        </label>
      </div>

      <p className={styles.caveat}>
        <b>Direction is real here.</b> Each row is one direction: the <b>environment</b> gene is
        expressed by the neighboring cells, the <b>cell</b> gene by the cell being modelled. A pair
        listed both ways has two independent coefficients — they are never merged. Sign is the
        claim: positive means the interaction raises that target gene, negative lowers it.
      </p>

      {/* Comparison bar: stack other channels below, aligned to this metabolite's target genes. */}
      <div className={styles.compare}>
        <span className={styles.compareLabel}>Compare</span>
        {added.map((id) => {
          const ch = channelsById.get(id);
          if (!ch) return null;
          const label = optionLabel(ch);
          return (
            <span key={id} className={styles.chip}>
              {label}
              <button
                type="button"
                className={styles.chipRemove}
                onClick={() => remove(id)}
                title={`Remove ${label}`}
                aria-label={`Remove ${label}`}
              >
                ✕
              </button>
            </span>
          );
        })}
        <select
          className="control"
          value=""
          aria-label="Add comparison"
          onChange={(e) => {
            const id = e.target.value as BetaChannelId;
            if (id) setAdded((a) => [...a, id]);
          }}
        >
          <option value="">Add comparison ▾</option>
          {availableToAdd.map((ch) => (
            <option key={ch.id} value={ch.id}>
              {optionLabel(ch)}
            </option>
          ))}
        </select>
      </div>

      {/* One cell-type-major matrix: columns are the UNION of target genes across shown groups, so
          a gene only a comparison factor covers still appears (with the metab group blank there). */}
      <BetaMatrix groups={groups} cellTypes={shown} />
    </section>
  );
}
