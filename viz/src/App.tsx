import { useEffect } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { rankGenePairs, rankMetabolites } from '@/data/ranking';
import ControlBar from '@/components/ControlBar';
import EntityPanel from '@/components/EntityPanel';
import Legend from '@/components/Legend';
import EntityDetails from '@/components/EntityDetails';
import EdgeDetails from '@/components/EdgeDetails';
import GenePairTabs from '@/components/GenePairTabs';
import NeighborhoodView from '@/components/NeighborhoodView';
import GraphView from '@/graph/GraphView';

export default function App() {
  const status = useVizStore((s) => s.status);
  const error = useVizStore((s) => s.error);
  const init = useVizStore((s) => s.init);

  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const edgeBundle = useVizStore((s) => s.edgeBundle);
  const selectEntity = useVizStore((s) => s.selectEntity);
  const view = useVizStore((s) => s.view);

  useEffect(() => {
    void init();
  }, [init]);

  // Auto-select the top-ranked entity of the CURRENT kind so the canvas is never blank on
  // arrival. Only fires when nothing is selected — switching kinds restores the remembered
  // entity first (store.entityByKind), and this fills in only when there is no memory yet.
  useEffect(() => {
    if (entityId || !dataset || !tier) return;
    if (entityKind === 'metabolite') {
      const ranked = rankMetabolites(dataset.entities.metabolite ?? [], tier.id);
      if (ranked[0]) selectEntity(ranked[0].entity.id);
      return;
    }
    // Gene-pair ranking is derived from the tier's edge bundle, so wait until the bundle for
    // THIS kind has landed — otherwise the "top" pair is just the first one alphabetically.
    if (edgeBundle?.entityKind !== 'gene_pair') return;
    const ranked = rankGenePairs(dataset.entities.gene_pair ?? [], edgeBundle.byEntity);
    if (ranked[0]) selectEntity(ranked[0].entity.id);
  }, [dataset, tier, entityKind, entityId, edgeBundle, selectEntity]);

  if (status === 'error') {
    return (
      <div className="empty" style={{ position: 'static', minHeight: '100vh' }}>
        <div>
          <p>Failed to load data.</p>
          <p className="muted">{error}</p>
          <p className="muted">
            Did you run <code>npm run ingest -- ../Results</code>?
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <ControlBar />
      <div className="app__body">
        <EntityPanel />
        <main className="canvas-wrap">
          {view === 'graph' ? (
            <>
              <GraphView />
              <GenePairTabs />
              <Legend />
              <EntityDetails />
              <EdgeDetails />
            </>
          ) : (
            <NeighborhoodView />
          )}
        </main>
      </div>
    </div>
  );
}
