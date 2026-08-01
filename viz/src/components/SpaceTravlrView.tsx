/**
 * Standalone SpaceTravLR view: the whole beta bundle, INDEPENDENT of the selected entity. A single
 * cell-type-major matrix — each cell type once, the selected factor channels (metab / lr / ltf / tf)
 * stacked beneath it, each on its OWN magnitude scale, with union target-gene columns.
 *
 * Controls: toggle channels (Sections), toggle target-gene columns (Target genes), and pick a
 * cell type. Toggles are tracked as EXCLUSION sets (empty = all shown), so a new tier/dataset never
 * needs re-selecting — new channels and new genes appear on by default.
 */
import { useMemo, useState } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import type { BetaChannelId } from '@/data/types';
import BetaMatrix, { type BetaFactorGroup } from './BetaMatrix';
import styles from './SpaceTravlrView.module.css';
import beta from './BetaPanel.module.css';

const ALL = '__all__';

export default function SpaceTravlrView() {
  const tier = useVizStore(selectCurrentTier);
  const betaBundle = useVizStore((s) => s.betaBundle);
  const bundleLoading = useVizStore((s) => s.bundleLoading);

  // Exclusion sets: empty = everything shown, so new channels/genes default ON across tiers.
  const [excluded, setExcluded] = useState<Set<BetaChannelId>>(new Set());
  const [excludedGenes, setExcludedGenes] = useState<Set<string>>(new Set());
  const [cellTypeFilter, setCellTypeFilter] = useState<string>(ALL);

  const channels = useMemo(() => betaBundle?.channels ?? [], [betaBundle]);
  const selectedChannels = useMemo(
    () => channels.filter((c) => !excluded.has(c.id)),
    [channels, excluded],
  );

  // Target genes: the union across the SELECTED channels (sorted), so hiding a channel drops the
  // genes only it contributed.
  const unionGenes = useMemo(
    () => [...new Set(selectedChannels.flatMap((c) => c.targetGenes))].sort(),
    [selectedChannels],
  );
  const selectedGenes = useMemo(
    () => unionGenes.filter((g) => !excludedGenes.has(g)),
    [unionGenes, excludedGenes],
  );

  // One factor group per selected channel (whole channel's rows), rendered cell-type-major.
  const groups = useMemo<BetaFactorGroup[]>(
    () => selectedChannels.map((c) => ({ key: c.id, channel: c, label: c.label, rows: c.rows })),
    [selectedChannels],
  );

  // Cell-type names differ per tier, so a filter set at Tier3 is meaningless at Tier1. Fall back
  // to "All" rather than blanking the view.
  const cellTypes = betaBundle?.cellTypes ?? [];
  const effectiveFilter = cellTypes.includes(cellTypeFilter) ? cellTypeFilter : ALL;
  const shown = effectiveFilter === ALL ? cellTypes : [effectiveFilter];

  const toggleChannel = (id: BetaChannelId) =>
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

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

        <div className={styles.group}>
          <span className={styles.groupLabel}>Sections</span>
          <div className={beta.chips} role="group" aria-label="Sections">
            {channels.map((c) => {
              const on = !excluded.has(c.id);
              return (
                <button
                  key={c.id}
                  type="button"
                  className={`${beta.chip} ${on ? beta.chipOn : ''}`}
                  aria-pressed={on}
                  onClick={() => toggleChannel(c.id)}
                >
                  {c.label}
                </button>
              );
            })}
          </div>
        </div>

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
        {selectedChannels.length === 0 ? (
          <div className={styles.none}>Select at least one section.</div>
        ) : selectedGenes.length === 0 ? (
          <div className={styles.none}>Select at least one target gene.</div>
        ) : (
          <BetaMatrix groups={groups} cellTypes={shown} allowedGenes={selectedGenes} />
        )}
      </div>
    </div>
  );
}
