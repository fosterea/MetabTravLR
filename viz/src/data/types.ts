/**
 * App-facing data contract.
 *
 * These types are the ONLY interface between the messy source outputs (harreman
 * CSV/JSON today, SpaceTravLR betadata tomorrow) and the UI. The ingest adapter
 * (`scripts/ingest.mjs`) is responsible for producing exactly these shapes; the
 * app never reads a raw CSV. Keep this file and the adapter in lockstep, and
 * mirror any change in `viz/docs/05_data_contract.md`.
 *
 * Design note — everything is "entity-agnostic": an edge connects two cell types
 * for some ENTITY, where an entity is a metabolite or a gene pair today, and
 * will be a gene / gene-set with a signed coefficient under SpaceTravLR. The
 * graph layer only knows about `EntityEdge`, not about metabolites specifically.
 */

/** Kinds of entity a graph can be keyed on. Extensible (future: 'gene', 'gene_set'). */
export type EntityKind = 'metabolite' | 'gene_pair';

/** Score bundle carried on every cell-type-pair edge (harreman NP + parametric). */
export interface EdgeScores {
  /** Parametric communication strength. */
  C_p: number;
  /** Parametric Z-score. */
  Z: number;
  /** Parametric FDR. */
  Z_FDR: number;
  /** Non-parametric communication strength (primary magnitude for the UI). */
  C_np: number;
  /** Non-parametric FDR (primary significance for the UI). */
  FDR_np: number;
  /** harreman's own significance call: FDR_np < thr AND C_np > 0. */
  selected: boolean;
}

/**
 * One undirected cell-type interface for a given entity.
 * IMPORTANT: `source`/`target` are NOT a flow direction — harreman's CT1→CT2 is a
 * sorted-label artifact (see docs/05_harreman_reference in the parent project).
 * `source === target` is a within-cell-type (self-loop / diagonal) edge.
 *
 * Forward-looking: `value`/`sign` are reserved for the SpaceTravLR two-value
 * encoding (magnitude → width, sign → diverging color). Unused for harreman.
 */
export interface EntityEdge {
  source: string;
  target: string;
  scores: EdgeScores;
  value?: number;
  sign?: -1 | 0 | 1;
}

/** A tier = one cell-type annotation granularity. Tiers form a parent hierarchy. */
export interface Tier {
  id: string; // "Tier1"
  label: string; // human label if we have one, else id
  cellTypes: string[];
  /** Coarser tier this one refines, or null for the coarsest. Powers "parent cell type". */
  parentTier: string | null;
  /**
   * Optional map child-cell-type -> parent-cell-type in `parentTier`.
   * Null until we have the annotation crosswalk; the UI must tolerate absence.
   */
  cellTypeParents?: Record<string, string> | null;
}

/** A metabolite entity + ranking metrics used by the side panel. */
export interface MetaboliteEntity {
  id: string;
  name: string;
  kind: 'metabolite';
  nGenePairs: number;
  genePairs: [string, string][];
  globalSignificant: boolean;
  globalFDR: number | null;
  nSigGenePairsGlobal: number | null;
  /** Per-tier summary metrics for ranking (keyed by tier id). */
  perTier: Record<string, MetaboliteTierSummary>;
}

export interface MetaboliteTierSummary {
  nSigPairs: number | null;
  tcellInvolved: boolean | null;
  withinTcell: string | null;
  tcellInterfaces: string | null;
  interactions: string | null;
}

/** A gene-pair entity. gp<->metabolite is many-to-many (a pair can serve several). */
export interface GenePairEntity {
  id: string; // "GENE1__GENE2"
  genes: [string, string];
  kind: 'gene_pair';
  metabolites: string[];
}

export type Entity = MetaboliteEntity | GenePairEntity;

/** Precomputed edges for one (tier, entityKind), indexed by entity id for O(1) lookup. */
export interface EdgeBundle {
  tier: string;
  entityKind: EntityKind;
  cellTypes: string[];
  byEntity: Record<string, EntityEdge[]>;
}

/** Per-dataset descriptor (`public/data/<id>/dataset.json`). */
export interface Dataset {
  id: string;
  name: string;
  source: 'harreman' | 'spacetravlr';
  entityKinds: EntityKind[];
  tiers: Tier[];
  entities: {
    metabolite?: MetaboliteEntity[];
    gene_pair?: GenePairEntity[];
  };
}

/** Top-level index of all datasets (`public/data/manifest.json`). */
export interface Manifest {
  /** ISO string stamped by the caller (ingest cannot read the clock). */
  generatedAt: string | null;
  schemaVersion: number;
  datasets: DatasetRef[];
}

export interface DatasetRef {
  id: string;
  name: string;
  source: 'harreman' | 'spacetravlr';
  entityKinds: EntityKind[];
  tierIds: string[];
}
