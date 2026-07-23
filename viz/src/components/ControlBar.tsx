/** Top control bar: dataset, tier, view, entity-kind, the FDR significance slider, and the
 *  show-non-significant toggle. */
import { Fragment, useMemo } from 'react';
import { useVizStore, hasEnvView } from '@/store/useVizStore';
import type { DatasetRef } from '@/data/types';

/** Discrete FDR_np cutoffs the significance slider steps through. Log-ish spaced over the range
 *  the non-parametric FDR actually occupies (its values are discrete with a ~0.003 floor), with
 *  the conventional 0.05 as the default. Note: harreman's per-tier gene-pair tables are
 *  significant-only (FDR < 0.05), so stops above 0.05 can only ever tighten, never reveal more
 *  gene-pair edges. */
const FDR_STOPS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2] as const;
/** Nearest stop index for a threshold value (always exact for slider-set values). */
const fdrIndex = (thr: number): number => {
  let best = 0;
  for (let i = 1; i < FDR_STOPS.length; i++) {
    if (Math.abs(FDR_STOPS[i] - thr) < Math.abs(FDR_STOPS[best] - thr)) best = i;
  }
  return best;
};

export default function ControlBar() {
  const manifest = useVizStore((s) => s.manifest);
  const dataset = useVizStore((s) => s.dataset);
  const datasetId = useVizStore((s) => s.datasetId);
  const tierId = useVizStore((s) => s.tierId);
  const entityKind = useVizStore((s) => s.entityKind);
  const showNonSignificant = useVizStore((s) => s.showNonSignificant);
  const fdrThreshold = useVizStore((s) => s.fdrThreshold);
  const gpExpandMode = useVizStore((s) => s.gpExpandMode);
  const gpExpandAll = useVizStore((s) => s.gpExpandAll);
  const gpTab = useVizStore((s) => s.gpTab);
  const view = useVizStore((s) => s.view);
  // These metabolite controls only apply to the graph, and not while a single gene-pair tab
  // is isolated.
  const metabControls = view === 'graph' && entityKind === 'metabolite' && !gpTab;

  const selectDataset = useVizStore((s) => s.selectDataset);
  const selectTier = useVizStore((s) => s.selectTier);
  const selectEntityKind = useVizStore((s) => s.selectEntityKind);
  const toggleNonSignificant = useVizStore((s) => s.toggleNonSignificant);
  const setFdrThreshold = useVizStore((s) => s.setFdrThreshold);
  const setGpExpandMode = useVizStore((s) => s.setGpExpandMode);
  const toggleGpExpandAll = useVizStore((s) => s.toggleGpExpandAll);
  const setView = useVizStore((s) => s.setView);

  // Group datasets by their source project folder, so a future second project stays legible.
  // Datasets whose harreman run never finished are listed but disabled, with the reason —
  // silently hiding them would look like data loss.
  const groups = useMemo(() => {
    const by = new Map<string, DatasetRef[]>();
    for (const d of manifest?.datasets ?? []) {
      const key = d.project ?? '';
      if (!by.has(key)) by.set(key, []);
      by.get(key)!.push(d);
    }
    return [...by.entries()];
  }, [manifest]);

  return (
    <header className="controlbar">
      <div className="controlbar__brand">
        MetabTravLR
        <small>crosstalk explorer</small>
      </div>

      <div className="field">
        <label htmlFor="dataset">Dataset</label>
        <select
          id="dataset"
          className="control"
          value={datasetId ?? ''}
          onChange={(e) => selectDataset(e.target.value)}
        >
          {groups.map(([project, ds]) => {
            const options = ds.map((d) => (
              <option
                key={d.id}
                value={d.id}
                disabled={d.available === false}
                title={d.unavailableReason}
              >
                {d.name}
                {d.available === false ? ' — incomplete run' : ''}
              </option>
            ));
            // Only wrap in a <optgroup> when there is something to distinguish.
            return project && groups.length > 1 ? (
              <optgroup key={project} label={project}>
                {options}
              </optgroup>
            ) : (
              <Fragment key={project || 'all'}>{options}</Fragment>
            );
          })}
        </select>
      </div>

      <div className="field">
        <label htmlFor="tier">Tier (cell-type resolution)</label>
        <select
          id="tier"
          className="control"
          value={tierId ?? ''}
          onChange={(e) => selectTier(e.target.value)}
        >
          {dataset?.tiers.map((t) => (
            <option key={t.id} value={t.id}>
              {t.label} · {t.cellTypes.length} types
            </option>
          ))}
        </select>
      </div>

      {/* Graph = harreman's cell-type INTERFACE statistic. Environment = everything measured on a
          cell against its own label rather than an interface: the per-cell neighborhood scores and
          the SpaceTravLR coefficients. A different question, hence a different view rather than
          another encoding on the same graph. */}
      <div className="field">
        <label>View</label>
        <div className="segmented" role="group" aria-label="View">
          <button aria-pressed={view === 'graph'} onClick={() => setView('graph')}>
            Graph
          </button>
          <button
            aria-pressed={view === 'nbhd'}
            onClick={() => setView('nbhd')}
            disabled={!hasEnvView(dataset)}
            title={
              hasEnvView(dataset)
                ? 'Neighborhood scores and SpaceTravLR coefficients for the selected entity'
                : 'This dataset has neither neighborhood scores nor SpaceTravLR coefficients'
            }
          >
            Environment
          </button>
        </div>
      </div>

      <div className="field">
        <label>Entity</label>
        <div className="segmented" role="group" aria-label="Entity kind">
          <button
            aria-pressed={entityKind === 'metabolite'}
            onClick={() => selectEntityKind('metabolite')}
          >
            Metabolite
          </button>
          <button
            aria-pressed={entityKind === 'gene_pair'}
            onClick={() => selectEntityKind('gene_pair')}
            title="Gene-pair view (transporter pairs)"
          >
            Gene pair
          </button>
        </div>
      </div>

      {/* Metabolite → transporter gene-pair expansion. "In panel" lists the pairs in the edge
          details; "On graph" fans the picked (or all) interface into gene-pair sub-edges. */}
      {metabControls && (
        <div className="field">
          <label>Gene pairs</label>
          <div className="segmented" role="group" aria-label="Gene-pair expansion">
            <button
              aria-pressed={gpExpandMode === 'panel'}
              onClick={() => setGpExpandMode('panel')}
              title="List a metabolite edge's transporter pairs in the details panel"
            >
              In panel
            </button>
            <button
              aria-pressed={gpExpandMode === 'graph'}
              onClick={() => setGpExpandMode('graph')}
              title="Fan a metabolite interface out into its transporter gene-pair sub-edges"
            >
              On graph
            </button>
          </div>
        </div>
      )}

      {metabControls && gpExpandMode === 'graph' && (
        <label className="checkbox">
          <input type="checkbox" checked={gpExpandAll} onChange={toggleGpExpandAll} />
          Expand all interfaces
        </label>
      )}

      {/* FDR-controlled non-parametric significance cutoff. Applies to both the metabolite and
          gene-pair graph views (not the Environment view). Discrete stops — the FDR_np values are
          themselves discrete with a low floor — defaulting to 0.05, which reproduces harreman's
          own significance call exactly. */}
      {view === 'graph' && (
        <div className="field" style={{ marginLeft: 'auto' }}>
          <label htmlFor="fdr">
            Significance · FDR &lt; {fdrThreshold}
          </label>
          <input
            id="fdr"
            type="range"
            min={0}
            max={FDR_STOPS.length - 1}
            step={1}
            value={fdrIndex(fdrThreshold)}
            onChange={(e) => setFdrThreshold(FDR_STOPS[Number(e.target.value)])}
            list="fdr-stops"
            style={{ width: 160 }}
            title="FDR-controlled non-parametric significance cutoff (harreman FDR_np). Steps 0.001 → 0.2; default 0.05."
          />
          <datalist id="fdr-stops">
            {FDR_STOPS.map((s, i) => (
              <option key={s} value={i} label={String(s)} />
            ))}
          </datalist>
        </div>
      )}

      {/* Gene-pair edges come from harreman's _gp_sig table (significant-only), so the
          non-significant toggle is meaningless there — only offer it for metabolites. */}
      {metabControls && (
        <label className="checkbox">
          <input type="checkbox" checked={showNonSignificant} onChange={toggleNonSignificant} />
          Show non-significant edges
        </label>
      )}
    </header>
  );
}
