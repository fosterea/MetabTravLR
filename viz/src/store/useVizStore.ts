/**
 * Global selection + loaded data. Zustand keeps the shared view state (dataset/tier/
 * entityKind/entity/toggles) in one place; components read via selectors and never fetch
 * data themselves. Async actions load the manifest, dataset descriptor, and the edge bundle
 * for the current (tier, entityKind).
 */
import { create } from 'zustand';
import type { Dataset, EdgeBundle, EntityKind, Manifest, Tier } from '@/data/types';
import { fetchDataset, fetchEdgeBundle, fetchManifest } from '@/data/loadDataset';

type Status = 'idle' | 'loading' | 'ready' | 'error';

interface VizState {
  status: Status;
  error?: string;

  manifest?: Manifest;
  datasetId?: string;
  dataset?: Dataset;

  tierId?: string;
  entityKind: EntityKind;
  entityId?: string;
  showNonSignificant: boolean;

  /** The clicked edge (an undirected cell-type interface), identified by its endpoints. */
  selectedEdge?: { source: string; target: string };

  /** How a metabolite edge's transporter gene pairs are revealed: listed in the details
   *  panel, or fanned out as parallel sub-edges on the graph. */
  gpExpandMode: 'panel' | 'graph';
  /** Graph mode only: fan out every interface at once (vs just the clicked one). */
  gpExpandAll: boolean;
  /** A single transporter gene pair (id) of the current metabolite, isolated in its own
   *  "tab": the graph shows only that pair's interfaces. Undefined = the metabolite ("All"). */
  gpTab?: string;

  /** Edge bundle for the current (datasetId, tierId, entityKind). */
  edgeBundle?: EdgeBundle;
  /** Same-tier gene_pair bundle, loaded alongside in metabolite mode for gp expansion. */
  gpBundle?: EdgeBundle;
  bundleLoading: boolean;

  init: () => Promise<void>;
  selectDataset: (id: string) => Promise<void>;
  selectTier: (tierId: string) => Promise<void>;
  selectEntityKind: (kind: EntityKind) => Promise<void>;
  selectEntity: (entityId: string | undefined) => void;
  selectEdge: (edge: { source: string; target: string } | undefined) => void;
  setGpExpandMode: (mode: 'panel' | 'graph') => void;
  toggleGpExpandAll: () => void;
  setGpTab: (gpTab: string | undefined) => void;
  toggleNonSignificant: () => void;
}

export const useVizStore = create<VizState>((set, get) => ({
  status: 'idle',
  entityKind: 'metabolite',
  showNonSignificant: false,
  gpExpandMode: 'panel',
  gpExpandAll: false,
  bundleLoading: false,

  init: async () => {
    if (get().status === 'loading') return;
    set({ status: 'loading', error: undefined });
    try {
      const manifest = await fetchManifest();
      set({ manifest });
      const first = manifest.datasets[0];
      if (!first) {
        set({ status: 'ready' });
        return;
      }
      await get().selectDataset(first.id);
      // selectDataset self-reports errors; don't mask them with 'ready'.
      if (get().status !== 'error') set({ status: 'ready' });
    } catch (e) {
      set({ status: 'error', error: (e as Error).message });
    }
  },

  selectDataset: async (id) => {
    try {
      const dataset = await fetchDataset(id);
      const tierId = dataset.tiers.at(-1)?.id ?? dataset.tiers[0]?.id; // default to finest tier
      set({
        datasetId: id,
        dataset,
        tierId,
        entityId: undefined,
        selectedEdge: undefined,
        edgeBundle: undefined,
        gpBundle: undefined,
        gpTab: undefined,
      });
      if (tierId) await loadBundle(set, get);
    } catch (e) {
      set({ status: 'error', error: (e as Error).message });
    }
  },

  selectTier: async (tierId) => {
    try {
      // Drop the stale bundle so the graph never renders a metabolite's old-tier edges.
      set({ tierId, selectedEdge: undefined, edgeBundle: undefined, gpBundle: undefined, gpTab: undefined });
      await loadBundle(set, get);
    } catch (e) {
      set({ status: 'error', error: (e as Error).message });
    }
  },

  selectEntityKind: async (kind) => {
    try {
      // Drop the stale bundle: metabolite edges must not linger into gene-pair mode.
      set({
        entityKind: kind,
        entityId: undefined,
        selectedEdge: undefined,
        edgeBundle: undefined,
        gpBundle: undefined,
        gpTab: undefined,
      });
      await loadBundle(set, get);
    } catch (e) {
      set({ status: 'error', error: (e as Error).message });
    }
  },

  // Changing the entity swaps the whole edge set, so any picked edge is now stale.
  selectEntity: (entityId) => set({ entityId, selectedEdge: undefined, gpTab: undefined }),

  selectEdge: (selectedEdge) => set({ selectedEdge }),

  setGpExpandMode: (gpExpandMode) => set({ gpExpandMode }),

  toggleGpExpandAll: () => set((s) => ({ gpExpandAll: !s.gpExpandAll })),

  // Isolating a gene-pair tab swaps the whole displayed edge set, so drop any picked edge.
  setGpTab: (gpTab) => set({ gpTab, selectedEdge: undefined }),

  toggleNonSignificant: () => set((s) => ({ showNonSignificant: !s.showNonSignificant })),
}));

async function loadBundle(set: (partial: Partial<VizState>) => void, get: () => VizState) {
  const { datasetId, tierId, entityKind } = get();
  if (!datasetId || !tierId) return;
  set({ bundleLoading: true });
  // The selection this request was issued for; ignore results if it changed underneath us.
  const matchesRequest = () => {
    const cur = get();
    return cur.datasetId === datasetId && cur.tierId === tierId && cur.entityKind === entityKind;
  };
  try {
    // In metabolite mode also load the same-tier gene_pair bundle so a metabolite edge can be
    // expanded into its contributing transporter pairs (client-side; no extra ingest).
    const [edgeBundle, gpBundle] = await Promise.all([
      fetchEdgeBundle(datasetId, tierId, entityKind),
      entityKind === 'metabolite'
        ? fetchEdgeBundle(datasetId, tierId, 'gene_pair')
        : Promise.resolve(undefined),
    ]);
    if (matchesRequest()) {
      set({ edgeBundle, gpBundle, bundleLoading: false });
    } else {
      // Lost the race to a newer selection; still clear our flag so it can't stick true.
      set({ bundleLoading: false });
    }
  } catch (e) {
    if (matchesRequest()) {
      set({ error: (e as Error).message, bundleLoading: false });
    } else {
      set({ bundleLoading: false });
    }
  }
}

/** Selector: the current tier object. */
export const selectCurrentTier = (s: VizState): Tier | undefined =>
  s.dataset?.tiers.find((t) => t.id === s.tierId);

/** Selector: the current dataset (typed convenience). */
export const selectDataset = (s: VizState): Dataset | undefined => s.dataset;
