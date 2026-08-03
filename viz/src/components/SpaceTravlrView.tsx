/**
 * Standalone SpaceTravLR view: the whole beta bundle, INDEPENDENT of the selected entity. A single
 * cell-type-major matrix — each cell type once, the selected factors (chosen via the FactorPicker)
 * stacked beneath it, each channel on its OWN magnitude scale, with union target-gene columns.
 *
 * Controls: a `FactorPicker` (whole-group buttons + a searchable add-a-specific-factor box + per-
 * channel selected-factor sections) chooses the ROWS; target-gene chips choose the COLUMNS; a
 * cell-type select scopes the blocks. The factor selection defaults to every factor of every
 * channel (populated on arrival) and re-seeds to that default whenever the bundle (tier/dataset)
 * changes, since feature keys are (a,b)-specific and don't carry across tiers.
 */
import { useEffect, useMemo, useState } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import BetaMatrix, { type BetaFactorGroup } from './BetaMatrix';
import FactorPicker from './FactorPicker';
import { channelFeatures, selectedRows, type FeatureKey } from '@/data/factorSelection';
import styles from './SpaceTravlrView.module.css';
import beta from './BetaPanel.module.css';

const ALL = '__all__';

export default function SpaceTravlrView() {
  const tier = useVizStore(selectCurrentTier);
  const betaBundle = useVizStore((s) => s.betaBundle);
  const bundleLoading = useVizStore((s) => s.bundleLoading);

  const [selected, setSelected] = useState<Set<FeatureKey>>(() => new Set());
  const [excludedGenes, setExcludedGenes] = useState<Set<string>>(new Set());
  const [cellTypeFilter, setCellTypeFilter] = useState<string>(ALL);

  const channels = useMemo(() => betaBundle?.channels ?? [], [betaBundle]);

  // Default: every factor of every channel selected (no blank canvas; the per-channel sections just
  // render collapsed "N selected" summaries). Re-seed whenever the bundle changes (tier/dataset).
  useEffect(() => {
    setSelected(new Set(channels.flatMap((ch) => channelFeatures(ch).map((f) => f.key))));
  }, [channels]);

  // One factor group per channel that contributes any rows under the current selection.
  const groups = useMemo<BetaFactorGroup[]>(
    () =>
      channels
        .map((ch) => ({ key: ch.id, channel: ch, label: ch.label, rows: selectedRows(ch, selected) }))
        .filter((g) => g.rows.length > 0),
    [channels, selected],
  );

  // Target genes = the union across the selected factors' rows (sorted); chips toggle columns.
  const unionGenes = useMemo(
    () => [...new Set(groups.flatMap((g) => g.rows.map((r) => r.gene)))].sort(),
    [groups],
  );
  const selectedGenes = useMemo(
    () => unionGenes.filter((g) => !excludedGenes.has(g)),
    [unionGenes, excludedGenes],
  );

  // Cell-type names differ per tier, so a filter set at Tier3 is meaningless at Tier1. Fall back
  // to "All" rather than blanking the view.
  const cellTypes = betaBundle?.cellTypes ?? [];
  const effectiveFilter = cellTypes.includes(cellTypeFilter) ? cellTypeFilter : ALL;
  const shown = effectiveFilter === ALL ? cellTypes : [effectiveFilter];

  const toggleGene = (g: string) =>
    setExcludedGenes((prev) => {
      const next = new Set(prev);
      if (next.has(g)) next.delete(g);
      else next.add(g);
      return next;
    });

  if (bundleLoading) return <div className="empty">Loading…</div>;
  if (!betaBundle) return <div className="empty">This dataset has no SpaceTravLR coefficients.</div>;

  return (
    <div className={styles.wrap}>
      <div className={styles.settings}>
        <div className={styles.head}>
          <div>
            <h2 className={styles.title}>SpaceTravLR coefficients</h2>
            <div className={styles.subtitle}>
              How much each feature moves the target genes, per cell type · {tier?.label}. Each
              channel is on its own magnitude scale.
            </div>
          </div>
          <label className={beta.filter}>
            <span className={beta.filterLabel}>Cell type</span>
            <select
              className="control"
              value={effectiveFilter}
              onChange={(e) => setCellTypeFilter(e.target.value)}
            >
              <option value={ALL}>All ({cellTypes.length})</option>
              {cellTypes.map((ct) => (
                <option key={ct} value={ct}>
                  {ct}
                </option>
              ))}
            </select>
          </label>
        </div>

        <FactorPicker channels={channels} selected={selected} onChange={setSelected} />

        {unionGenes.length > 0 && (
          <div className={styles.group}>
            <span className={styles.groupLabel}>Target genes</span>
            <div className={beta.chips} role="group" aria-label="Target genes">
              {unionGenes.map((g) => {
                const on = !excludedGenes.has(g);
                return (
                  <button
                    key={g}
                    type="button"
                    className={`${beta.chip} ${on ? beta.chipOn : ''}`}
                    aria-pressed={on}
                    onClick={() => toggleGene(g)}
                  >
                    {g}
                  </button>
                );
              })}
              <button type="button" className={styles.quick} onClick={() => setExcludedGenes(new Set())}>
                All
              </button>
              <button
                type="button"
                className={styles.quick}
                onClick={() => setExcludedGenes(new Set(unionGenes))}
              >
                None
              </button>
            </div>
          </div>
        )}
      </div>

      <div className={styles.body}>
        {groups.length === 0 ? (
          <div className={styles.none}>Select at least one factor.</div>
        ) : selectedGenes.length === 0 ? (
          <div className={styles.none}>Select at least one target gene.</div>
        ) : (
          <BetaMatrix groups={groups} cellTypes={shown} allowedGenes={selectedGenes} />
        )}
      </div>
    </div>
  );
}
