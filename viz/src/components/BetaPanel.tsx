/**
 * SpaceTravLR coefficients for the selected entity, as a cell-type-major heatmap (`BetaMatrix`).
 *
 * The primary anchor group is the metabolite's (or gene pair's) own transporter pairs, always shown
 * first. Beneath it, a `FactorPicker` lets the user stack COMPARISON factors — whole channels
 * (ligand–receptor, ligand–TF, transcription factors, all metabolic transporters) and/or individual
 * factors (a single L–R pair, a single TF, a specific transporter pair). All groups are stacked
 * cell-type-major, sharing one UNION set of target-gene columns (so a gene only a comparison factor
 * covers still appears), while each channel keeps its OWN magnitude scale. The comparison selection
 * resets when the entity/tier/dataset changes, but survives a cell-type-filter change.
 *
 * Both directions of a pair are separate rows on purpose: `A→B` and `B→A` are different
 * coefficients, and merging them would invent a symmetry the model does not claim.
 */
import { useEffect, useMemo, useState } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { betaKey } from '@/data/betaScale';
import type { Entity } from '@/data/types';
import BetaMatrix, { type BetaFactorGroup } from './BetaMatrix';
import FactorPicker from './FactorPicker';
import { selectedRows, type FeatureKey } from '@/data/factorSelection';
import styles from './BetaPanel.module.css';

const ALL = '__all__';

/** The pair keys whose coefficients belong to this entity. */
function pairKeysFor(entity: Entity): string[] {
  if (entity.kind === 'gene_pair') return [betaKey(entity.genes[0], entity.genes[1])];
  // A metabolite is served by many transporter pairs (many-to-many); show all of them.
  return [...new Set(entity.genePairs.map(([a, b]) => betaKey(a, b)))];
}

export default function BetaPanel() {
  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const betaBundle = useVizStore((s) => s.betaBundle);
  const [cellTypeFilter, setCellTypeFilter] = useState<string>(ALL);

  // Comparison factors stacked below the primary anchor group. Empty by default — the panel is
  // about THIS metabolite; comparisons are a deliberate opt-in.
  const [selected, setSelected] = useState<Set<FeatureKey>>(() => new Set());
  // Stale comparisons must not linger onto a different entity/tier/dataset (their (a,b)/scales are no
  // longer meaningful). A cell-type-filter change deliberately does NOT reset them.
  const datasetId = dataset?.id;
  const tierId = tier?.id;
  useEffect(() => {
    setSelected(new Set());
  }, [entityId, tierId, datasetId]);

  const entity = useMemo<Entity | undefined>(() => {
    if (!dataset || !entityId) return undefined;
    const list =
      entityKind === 'metabolite' ? dataset.entities.metabolite : dataset.entities.gene_pair;
    return list?.find((e) => e.id === entityId);
  }, [dataset, entityId, entityKind]);

  const channels = useMemo(() => betaBundle?.channels ?? [], [betaBundle]);
  const metab = useMemo(() => channels.find((c) => c.id === 'metab'), [channels]);

  // This entity's own transporter-pair rows, filtered out of the metab channel by sorted pair key.
  const primaryRows = useMemo(() => {
    if (!entity || !metab) return [];
    const keys = new Set(pairKeysFor(entity));
    return metab.rows.filter((r) => r.b != null && keys.has(betaKey(r.a, r.b)));
  }, [entity, metab]);

  // The stacked factor groups: the metabolite's own pairs first (the anchor), then one group per
  // channel that contributes rows under the comparison selection.
  const groups = useMemo<BetaFactorGroup[]>(() => {
    if (!metab) return [];
    const primaryLabel = `This ${entityKind === 'metabolite' ? 'metabolite' : 'gene pair'}’s transporter pairs`;
    const primary: BetaFactorGroup = {
      key: 'metab-primary',
      channel: metab,
      label: primaryLabel,
      rows: primaryRows,
    };
    const comparisons = channels
      .map((ch): BetaFactorGroup => ({
        key: ch.id,
        channel: ch,
        label: ch.label,
        rows: selectedRows(ch, selected),
      }))
      .filter((g) => g.rows.length > 0);
    return [primary, ...comparisons];
  }, [metab, channels, entityKind, primaryRows, selected]);

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

      {/* Compare the metabolite's transporters against whole groups and/or individual factors. */}
      <FactorPicker channels={channels} selected={selected} onChange={setSelected} />

      {/* One cell-type-major matrix: columns are the UNION of target genes across shown groups, so
          a gene only a comparison factor covers still appears (with other groups blank there). */}
      <BetaMatrix groups={groups} cellTypes={shown} />
    </section>
  );
}
