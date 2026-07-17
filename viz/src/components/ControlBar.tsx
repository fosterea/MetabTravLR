/** Top control bar: dataset, tier, entity-kind, and the show-non-significant toggle. */
import { useVizStore } from '@/store/useVizStore';

export default function ControlBar() {
  const manifest = useVizStore((s) => s.manifest);
  const dataset = useVizStore((s) => s.dataset);
  const datasetId = useVizStore((s) => s.datasetId);
  const tierId = useVizStore((s) => s.tierId);
  const entityKind = useVizStore((s) => s.entityKind);
  const showNonSignificant = useVizStore((s) => s.showNonSignificant);

  const selectDataset = useVizStore((s) => s.selectDataset);
  const selectTier = useVizStore((s) => s.selectTier);
  const selectEntityKind = useVizStore((s) => s.selectEntityKind);
  const toggleNonSignificant = useVizStore((s) => s.toggleNonSignificant);

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
