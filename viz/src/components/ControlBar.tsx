/** Top control bar: dataset, tier, entity-kind, and the show-non-significant toggle. */
import { useVizStore } from '@/store/useVizStore';

export default function ControlBar() {
  const manifest = useVizStore((s) => s.manifest);
  const dataset = useVizStore((s) => s.dataset);
  const datasetId = useVizStore((s) => s.datasetId);
  const tierId = useVizStore((s) => s.tierId);
  const entityKind = useVizStore((s) => s.entityKind);
  const showNonSignificant = useVizStore((s) => s.showNonSignificant);
  const gpExpandMode = useVizStore((s) => s.gpExpandMode);
  const gpExpandAll = useVizStore((s) => s.gpExpandAll);

  const selectDataset = useVizStore((s) => s.selectDataset);
  const selectTier = useVizStore((s) => s.selectTier);
  const selectEntityKind = useVizStore((s) => s.selectEntityKind);
  const toggleNonSignificant = useVizStore((s) => s.toggleNonSignificant);
  const setGpExpandMode = useVizStore((s) => s.setGpExpandMode);
  const toggleGpExpandAll = useVizStore((s) => s.toggleGpExpandAll);

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
          {manifest?.datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
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
      {entityKind === 'metabolite' && (
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

      {entityKind === 'metabolite' && gpExpandMode === 'graph' && (
        <label className="checkbox">
          <input type="checkbox" checked={gpExpandAll} onChange={toggleGpExpandAll} />
          Expand all interfaces
        </label>
      )}

      {/* Gene-pair edges come from harreman's _gp_sig table (significant-only), so the
          non-significant toggle is meaningless there — only offer it for metabolites. */}
      {entityKind === 'metabolite' && (
        <label className="checkbox" style={{ marginLeft: 'auto' }}>
          <input type="checkbox" checked={showNonSignificant} onChange={toggleNonSignificant} />
          Show non-significant edges
        </label>
      )}
    </header>
  );
}
