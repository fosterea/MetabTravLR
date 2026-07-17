import { useEffect } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { rankMetabolites } from '@/data/ranking';
import ControlBar from '@/components/ControlBar';
import EntityPanel from '@/components/EntityPanel';
import Legend from '@/components/Legend';
import EntityDetails from '@/components/EntityDetails';
import GraphView from '@/graph/GraphView';

export default function App() {
  const status = useVizStore((s) => s.status);
  const error = useVizStore((s) => s.error);
  const init = useVizStore((s) => s.init);

  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const selectEntity = useVizStore((s) => s.selectEntity);

  useEffect(() => {
    void init();
  }, [init]);

  // Auto-select the top-ranked metabolite so the graph appears on first load.
  useEffect(() => {
    if (entityKind !== 'metabolite' || entityId || !dataset || !tier) return;
    const ranked = rankMetabolites(dataset.entities.metabolite ?? [], tier.id);
    if (ranked[0]) selectEntity(ranked[0].entity.id);
  }, [dataset, tier, entityKind, entityId, selectEntity]);

  if (status === 'error') {
    return (
      <div className="empty" style={{ position: 'static', minHeight: '100vh' }}>
        <div>
          <p>Failed to load data.</p>
          <p className="muted">{error}</p>
          <p className="muted">
            Did you run <code>npm run ingest -- ../easy_download --id harreman</code>?
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
          <GraphView />
          <Legend />
          <EntityDetails />
        </main>
      </div>
    </div>
  );
}
