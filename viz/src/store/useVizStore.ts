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

  /** Edge bundle for the current (datasetId, tierId, entityKind). */
  edgeBundle?: EdgeBundle;
  bundleLoading: boolean;

  init: () => Promise<void>;
  selectDataset: (id: string) => Promise<void>;
  selectTier: (tierId: string) => Promise<void>;
  selectEntityKind: (kind: EntityKind) => Promise<void>;
  selectEntity: (entityId: string | undefined) => void;
  toggleNonSignificant: () => void;
}

export const useVizStore = create<VizState>((set, get) => ({
  status: 'idle',
  entityKind: 'metabolite',
  showNonSignificant: false,
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
      set({ status: 'ready' });
    } catch (e) {
      set({ status: 'error', error: (e as Error).message });
    }
  },

  selectDataset: async (id) => {
    const dataset = await fetchDataset(id);
    const tierId = dataset.tiers.at(-1)?.id ?? dataset.tiers[0]?.id; // default to finest tier
    set({ datasetId: id, dataset, tierId, entityId: undefined, edgeBundle: undefined });
    if (tierId) await loadBundle(set, get);
  },

  selectTier: async (tierId) => {
    set({ tierId });
    await loadBundle(set, get);
  },

  selectEntityKind: async (kind) => {
    set({ entityKind: kind, entityId: undefined });
    await loadBundle(set, get);
  },

  selectEntity: (entityId) => set({ entityId }),

  toggleNonSignificant: () => set((s) => ({ showNonSignificant: !s.showNonSignificant })),
}));

async function loadBundle(set: (partial: Partial<VizState>) => void, get: () => VizState) {
  const { datasetId, tierId, entityKind } = get();
  if (!datasetId || !tierId) return;
  set({ bundleLoading: true });
  try {
    const edgeBundle = await fetchEdgeBundle(datasetId, tierId, entityKind);
    // Ignore if selection changed while loading.
    const cur = get();
    if (cur.datasetId === datasetId && cur.tierId === tierId && cur.entityKind === entityKind) {
      set({ edgeBundle, bundleLoading: false });
    }
  } catch (e) {
    set({ error: (e as Error).message, bundleLoading: false });
  }
}

/** Selector: the current tier object. */
export const selectCurrentTier = (s: VizState): Tier | undefined =>
  s.dataset?.tiers.find((t) => t.id === s.tierId);

/** Selector: the current dataset (typed convenience). */
export const selectDataset = (s: VizState): Dataset | undefined => s.dataset;
