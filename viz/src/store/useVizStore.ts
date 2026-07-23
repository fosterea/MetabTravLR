/**
 * Global selection + loaded data. Zustand keeps the shared view state (dataset/tier/
 * entityKind/entity/toggles) in one place; components read via selectors and never fetch
 * data themselves. Async actions load the manifest, dataset descriptor, and the edge bundle
 * for the current (tier, entityKind).
 */
import { create } from 'zustand';
import type {
  BetaBundle,
  Dataset,
  EdgeBundle,
  EntityKind,
  Manifest,
  NbhdBundle,
  Tier,
} from '@/data/types';
import {
  fetchBetaBundle,
  fetchDataset,
  fetchEdgeBundle,
  fetchManifest,
  fetchNbhdBundle,
} from '@/data/loadDataset';

type Status = 'idle' | 'loading' | 'ready' | 'error';

/** Which visualization occupies the canvas. The side panel + entity selection are shared, so
 *  switching views keeps your place. */
export type View = 'graph' | 'nbhd';

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
  /** FDR_np cutoff for calling an interface significant. Default 0.05 (harreman's own default,
   *  which reproduces `scores.selected`). A global exploration preference — NOT reset on
   *  dataset/tier/kind change, exactly like `showNonSignificant`. */
  fdrThreshold: number;
  view: View;

  /**
   * Last entity selected in each kind, so switching Metabolite ⇄ Gene pair (and back) returns
   * you to where you were instead of resetting. Cleared when the dataset changes, since ids
   * are only meaningful within a dataset.
   */
  entityByKind: Partial<Record<EntityKind, string>>;

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
  /**
   * Gene-pair highlight, shared BOTH ways between the graph's fanned sub-edges and the details
   * panel's pair list: clicking a sub-edge pins its panel row, hovering either side previews.
   * Kept as two fields so a hover preview can end without destroying the pinned choice —
   * read them through `selectFocusedGp`, never directly.
   */
  pinnedGp?: string;
  hoverGp?: string;

  /** Edge bundle for the current (datasetId, tierId, entityKind). */
  edgeBundle?: EdgeBundle;
  /** Same-tier gene_pair bundle, loaded alongside in metabolite mode for gp expansion. */
  gpBundle?: EdgeBundle;
  /** Neighborhood scores for the current (datasetId, tierId) — kind-independent. */
  nbhdBundle?: NbhdBundle;
  /** SpaceTravLR coefficients for the current (datasetId, tierId) — kind-independent. */
  betaBundle?: BetaBundle;
  bundleLoading: boolean;

  init: () => Promise<void>;
  selectDataset: (id: string) => Promise<void>;
  selectTier: (tierId: string) => Promise<void>;
  selectEntityKind: (kind: EntityKind) => Promise<void>;
  selectEntity: (entityId: string | undefined) => void;
  /** Cross-navigate: jump to an entity of a possibly different kind (panel links). */
  goToEntity: (kind: EntityKind, entityId: string) => Promise<void>;
  selectEdge: (edge: { source: string; target: string } | undefined) => void;
  setGpExpandMode: (mode: 'panel' | 'graph') => void;
  toggleGpExpandAll: () => void;
  setGpTab: (gpTab: string | undefined) => void;
  /** Pin a gene pair (click). Passing the pinned id again unpins it. */
  setPinnedGp: (id: string | undefined) => void;
  /** Preview a gene pair (hover); undefined ends the preview. */
  setHoverGp: (id: string | undefined) => void;
  toggleNonSignificant: () => void;
  setFdrThreshold: (threshold: number) => void;
  setView: (view: View) => void;
}

export const useVizStore = create<VizState>((set, get) => ({
  status: 'idle',
  entityKind: 'metabolite',
  showNonSignificant: false,
  fdrThreshold: 0.05,
  gpExpandMode: 'panel',
  gpExpandAll: false,
  bundleLoading: false,
  view: 'graph',
  entityByKind: {},

  init: async () => {
    if (get().status === 'loading') return;
    set({ status: 'loading', error: undefined });
    try {
      const manifest = await fetchManifest();
      set({ manifest });
      // Incomplete runs have no files on disk; never auto-select one.
      const first = manifest.datasets.find((d) => d.available !== false);
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
    // Guard: an unavailable dataset has no dataset.json — loading it would 404 into the error
    // screen. The control bar disables these, so this is belt-and-braces.
    const ref = get().manifest?.datasets.find((d) => d.id === id);
    if (ref && ref.available === false) return;
    try {
      const dataset = await fetchDataset(id);
      const tierId = dataset.tiers.at(-1)?.id ?? dataset.tiers[0]?.id; // default to finest tier
      set({
        datasetId: id,
        dataset,
        tierId,
        entityId: undefined,
        entityByKind: {}, // entity ids are dataset-scoped
        selectedEdge: undefined,
        edgeBundle: undefined,
        gpBundle: undefined,
        nbhdBundle: undefined,
        betaBundle: undefined,
        gpTab: undefined,
        pinnedGp: undefined,
        hoverGp: undefined,
        // A dataset with neither neighborhood scores nor SpaceTravLR betas has nothing to put
        // in that view.
        view: hasEnvView(dataset) ? get().view : 'graph',
      });
      if (tierId) await loadBundle(set, get);
    } catch (e) {
      set({ status: 'error', error: (e as Error).message });
    }
  },

  selectTier: async (tierId) => {
    try {
      // Drop the stale bundle so the graph never renders a metabolite's old-tier edges.
      set({
        tierId,
        selectedEdge: undefined,
        edgeBundle: undefined,
        gpBundle: undefined,
        nbhdBundle: undefined,
        betaBundle: undefined,
        gpTab: undefined,
        pinnedGp: undefined,
        hoverGp: undefined,
      });
      await loadBundle(set, get);
    } catch (e) {
      set({ status: 'error', error: (e as Error).message });
    }
  },

  selectEntityKind: async (kind) => {
    try {
      const s = get();
      if (s.entityKind === kind) return;
      // Remember where we were in the outgoing kind, and restore where we were in the incoming
      // one (undefined ⇒ App auto-selects that kind's top-ranked entity).
      const entityByKind = { ...s.entityByKind };
      if (s.entityId) entityByKind[s.entityKind] = s.entityId;
      // Drop the stale bundle: metabolite edges must not linger into gene-pair mode.
      set({
        entityKind: kind,
        entityByKind,
        entityId: entityByKind[kind],
        selectedEdge: undefined,
        edgeBundle: undefined,
        gpBundle: undefined,
        gpTab: undefined,
        pinnedGp: undefined,
        hoverGp: undefined,
      });
      await loadBundle(set, get);
    } catch (e) {
      set({ status: 'error', error: (e as Error).message });
    }
  },

  // Changing the entity swaps the whole edge set, so any picked edge is now stale.
  selectEntity: (entityId) =>
    set((s) => ({
      entityId,
      entityByKind: entityId ? { ...s.entityByKind, [s.entityKind]: entityId } : s.entityByKind,
      selectedEdge: undefined,
      gpTab: undefined,
      pinnedGp: undefined,
      hoverGp: undefined,
    })),

  goToEntity: async (kind, entityId) => {
    if (get().entityKind !== kind) await get().selectEntityKind(kind);
    get().selectEntity(entityId);
  },

  selectEdge: (selectedEdge) => set({ selectedEdge }),

  setGpExpandMode: (gpExpandMode) => set({ gpExpandMode }),

  toggleGpExpandAll: () => set((s) => ({ gpExpandAll: !s.gpExpandAll })),

  // Isolating a gene-pair tab swaps the whole displayed edge set, so drop any picked edge.
  setGpTab: (gpTab) =>
    set({ gpTab, selectedEdge: undefined, pinnedGp: undefined, hoverGp: undefined }),

  setPinnedGp: (id) => set((s) => ({ pinnedGp: s.pinnedGp === id ? undefined : id })),

  setHoverGp: (hoverGp) => set({ hoverGp }),

  toggleNonSignificant: () => set((s) => ({ showNonSignificant: !s.showNonSignificant })),

  setFdrThreshold: (fdrThreshold) => set({ fdrThreshold }),

  setView: (view) => set({ view }),
}));

async function loadBundle(set: (partial: Partial<VizState>) => void, get: () => VizState) {
  const { datasetId, tierId, entityKind, dataset } = get();
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
    const [edgeBundle, gpBundle, nbhdBundle, betaBundle] = await Promise.all([
      fetchEdgeBundle(datasetId, tierId, entityKind),
      entityKind === 'metabolite'
        ? fetchEdgeBundle(datasetId, tierId, 'gene_pair')
        : Promise.resolve(undefined),
      // Neighborhood scores are kind-independent, so only (re)fetch when the tier changed.
      // A dataset from before the nbhd wrapper simply has none.
      dataset?.hasNbhd && get().nbhdBundle?.tier !== tierId
        ? fetchNbhdBundle(datasetId, tierId)
        : Promise.resolve(get().nbhdBundle),
      // Same deal for the SpaceTravLR coefficients: keyed on (dataset, tier) only.
      dataset?.hasBeta && get().betaBundle?.tier !== tierId
        ? fetchBetaBundle(datasetId, tierId)
        : Promise.resolve(get().betaBundle),
    ]);
    if (matchesRequest()) {
      set({ edgeBundle, gpBundle, nbhdBundle, betaBundle, bundleLoading: false });
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

/**
 * Does this dataset have anything to show in the environment view? Either harreman neighborhood
 * scores or SpaceTravLR coefficients is enough — the view renders whichever sections it has.
 */
export const hasEnvView = (d: Dataset | { hasNbhd: boolean; hasBeta: boolean } | undefined) =>
  Boolean(d && (d.hasNbhd || d.hasBeta));

/** Selector: the gene pair to highlight — a live hover preview wins over the pinned choice. */
export const selectFocusedGp = (s: VizState): string | undefined => s.hoverGp ?? s.pinnedGp;

/** Selector: the current tier object. */
export const selectCurrentTier = (s: VizState): Tier | undefined =>
  s.dataset?.tiers.find((t) => t.id === s.tierId);

/** Selector: the current dataset (typed convenience). */
export const selectDataset = (s: VizState): Dataset | undefined => s.dataset;
