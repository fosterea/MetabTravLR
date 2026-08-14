/**
 * SpaceTravLR coefficients for the selected entity, as a cell-type-major heatmap (`BetaMatrix`).
 *
 * The primary anchor group is the selected METABOLITE's own coefficients, always shown first —
 * sourced from the single-member `metab` channel (`metabolites.csv`, one row per whole metabolite,
 * summed over its transporters). A metabolite links to a row when the entity name equals the row's
 * `metabolite` value OR is one of the pipe-split members of a grouped value (metabolites that share
 * a transporter set are reported together, sharing one coefficient). Gene-pair entities have no
 * metabolite to anchor to, so their primary group is empty and the "no coefficients" empty-state
 * shows.
 *
 * Beneath the primary, a `FactorPicker` lets the user stack COMPARISON factors — whole channels
 * (metabolites, ligand–receptor, ligand–TF, transcription factors) and/or individual factors. All
 * groups are stacked cell-type-major, sharing one UNION set of target-gene columns, while each
 * channel keeps its OWN magnitude scale. The comparison selection resets when the entity/tier/
 * dataset changes, but survives a cell-type-filter change.
 */
import { useEffect, useMemo, useState } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import type { Entity } from '@/data/types';
import BetaMatrix, { type BetaFactorGroup } from './BetaMatrix';
import FactorPicker from './FactorPicker';
import { selectedRows, type FeatureKey } from '@/data/factorSelection';
import styles from './BetaPanel.module.css';

const ALL = '__all__';

/**
 * A metab row's `metabolite` value matches an entity when it equals the entity name, or the entity
 * name is one of the pipe-split members of a grouped value (`Adenosine|Cytidine|…`).
 */
function metaboliteMatches(rowMetab: string, entityName: string): boolean {
  return rowMetab === entityName || rowMetab.split('|').includes(entityName);
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

  // This metabolite's own coefficients: the metab channel's single-member rows whose `metabolite`
  // (member `a`) matches the selected entity by exact-or-member. Only metabolite-kind entities can
  // match; a gene pair has no metabolite value, so it falls through to the empty-state.
  const primaryRows = useMemo(() => {
    if (!entity || !metab || entity.kind !== 'metabolite') return [];
    return metab.rows.filter((r) => metaboliteMatches(r.a, entity.name));
  }, [entity, metab]);

  // When the matched rows come from a pipe-GROUP value, surface which metabolites are grouped: they
  // share a transporter set and can't be distinguished, so they share one coefficient.
  // Assumes every matched row shares ONE metabolite value (true for the current data: no member name
  // maps to more than one distinct value). If a member ever appeared both standalone and in a group,
  // this note would reflect only the first value while the matrix would render both.
  const groupMembers = useMemo(() => {
    const value = primaryRows[0]?.a;
    return value && value.includes('|') ? value.split('|') : null;
  }, [primaryRows]);

  // The stacked factor groups: the metabolite's own coefficients first (the anchor), then one group
  // per channel that contributes rows under the comparison selection.
  const groups = useMemo<BetaFactorGroup[]>(() => {
    if (!metab) return [];
    const primary: BetaFactorGroup = {
      key: 'metab-primary',
      channel: metab,
      label: 'This metabolite',
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
  }, [metab, channels, primaryRows, selected]);

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
          {entityKind === 'metabolite' ? 'metabolite' : 'gene pair'} at {tier?.label}.
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
            How much this metabolite’s transport moves the target genes, per cell type
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
        Each value is how much this metabolite’s transport moves the target gene in cells of that
        type. <b>Sign is the claim:</b> positive raises that gene, negative lowers it. The metabolite
        is summed over its transporters, so <b>no direction is asserted</b> here.
      </p>

      {groupMembers && (
        <p className={styles.groupNote}>
          <b>Reported as a group:</b> {groupMembers.join(', ')}. These metabolites share the same
          transporter set and can’t be distinguished, so they share one coefficient.
        </p>
      )}

      {/* Compare this metabolite against whole channels and/or individual factors. */}
      <FactorPicker channels={channels} selected={selected} onChange={setSelected} />

      {/* One cell-type-major matrix: columns are the UNION of target genes across shown groups, so
          a gene only a comparison factor covers still appears (with other groups blank there). */}
      <BetaMatrix groups={groups} cellTypes={shown} />
    </section>
  );
}
