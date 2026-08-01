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

/**
 * One row of the **neighborhood scores** (`<Tier>/[nbhd_scores][summary_*].csv`): how much a
 * cell type's OWN cells sit in high-scoring neighborhoods for an entity.
 *
 * ⚠️ This is NOT the cell-type-interface statistic the graph draws. Each cell's score is bucketed
 * by that cell's own label, so a row reads "cells of type X score high on metabolite M" — never
 * "X exchanges M with Y", and it carries no direction. Keep it out of the edge model entirely.
 * `log2Enrichment` is unstable for tiny cell types; always weigh it against `nCells`.
 */
export interface NbhdRow {
  cellType: string;
  /** Cells of this type at this tier (the denominator; small ⇒ distrust log2Enrichment). */
  nCells: number | null;
  /** Fraction of those cells with a significant score for this entity. Primary UI magnitude. */
  fracSig: number | null;
  /** Mean score over all cells of the type. */
  meanCs: number | null;
  /** Mean score over only the significant cells. */
  meanCsSig: number | null;
  meanNegLog10P: number | null;
  /** log2(observed / expected) significant share. Unstable for small `nCells`. */
  log2Enrichment: number | null;
}

/** Neighborhood scores for one tier (`public/data/<id>/nbhd/<Tier>.json`). */
export interface NbhdBundle {
  tier: string;
  cellTypes: string[];
  byEntity: Record<EntityKind, Record<string, NbhdRow[]>>;
}

/**
 * The four SpaceTravLR feature channels. Each has its OWN target-gene set and its OWN magnitude
 * scale — TF means are order ~1e0, metab ~1e-6, LR ~1e-7, L-TF ~1e-9..1e-7 — so any encoding must
 * scale per channel (a single shared scale would render whole channels as ≈0). See `betaScale.ts`.
 */
export type BetaChannelId = 'metab' | 'lr' | 'ltf' | 'tf';

/**
 * One SpaceTravLR learned coefficient, generalized across all feature channels
 * (`metabtravlr_outputs/<Tier>/{gene_pairs,ligand_receptor,ligand_tf,transcription_factor}.csv`).
 * A row is: how much one feature moves one target gene, averaged over the cells of one cell type.
 *
 * ⚠️ Unlike `EntityEdge`, direction here is REAL and load-bearing. Member `a` is the
 * environment/sender side (export / ligand / ligand), OR the TF for a single-member channel;
 * member `b` is the cell/receiver side (import / receptor / tf), and is null for single-member
 * channels (tf). For transporter pairs both orders can be present as distinct rows and must never
 * be merged, averaged, or displayed as one row.
 *
 * `mean` is signed — the sign is the biological claim (up- vs down-regulation of `gene`) and is
 * the one thing an encoding must never distort. `std`/`n` are the spread and the cell count
 * behind the mean; no significance test is applied anywhere in this pipeline.
 */
export interface BetaRow {
  /** Feature member 0 = environment/sender side (export / ligand / ligand), OR the TF for a single-member channel. */
  a: string;
  /** Feature member 1 = cell/receiver side (import / receptor / tf); null for single-member channels (tf). */
  b: string | null;
  /** The target gene whose expression this coefficient moves. */
  gene: string;
  cellType: string;
  /** Signed mean coefficient. */
  mean: number | null;
  /** Standard deviation of the coefficient across the cells of this type. */
  std: number | null;
  /** Cells of this type behind the mean. */
  n: number | null;
}

/**
 * One feature channel (metab/lr/ltf/tf) for one tier. Self-describing so components don't hardcode
 * per-channel meta (labels, member roles, column headers) — they read it off the channel.
 */
export interface BetaChannel {
  id: BetaChannelId;
  /** "Metabolic transporters" | "Ligand–receptor" | "Ligand–TF" | "Transcription factors". */
  label: string;
  /** Pair channels have two members (a → b); single channels (tf) have only member `a`. */
  kind: 'pair' | 'single';
  /**
   * Role labels for tooltips, in member order:
   * `['export (environment)','import (cell)']`, `['ligand','receptor']`, `['ligand','TF']`, `['TF']`.
   */
  memberLabels: string[];
  /** Row-identity column header: "environment → cell" | "ligand → receptor" | "ligand → TF" | "transcription factor". */
  rowHeader: string;
  /** Target genes present in THIS channel, sorted. The channel heatmap's columns. */
  targetGenes: string[];
  /** Cell types present in THIS channel, in source order. */
  cellTypes: string[];
  /** All rows for this channel/tier, sorted strongest |mean| first. */
  rows: BetaRow[];
}

/**
 * SpaceTravLR coefficients for one tier (`public/data/<id>/beta/<Tier>.json`).
 *
 * Holds ALL feature channels (v4). The metabolite lookup filters the `metab` channel's rows by
 * `betaKey(r.a, r.b) ∈ pairKeys` (see `betaScale.ts`) — there is no `byPair` index anymore, since
 * a directed row can't be attributed to a single ordered entity id.
 */
export interface BetaBundle {
  tier: string;
  /** Union of cell types across channels, in source order. */
  cellTypes: string[];
  /** Union of target genes across channels, sorted. */
  targetGenes: string[];
  /** Only channels with ≥1 row are present. */
  channels: BetaChannel[];
}

/** Per-dataset descriptor (`public/data/<id>/dataset.json`). */
export interface Dataset {
  id: string;
  name: string;
  /** Grouping folder the dataset came from (`Results/<project>/<dataset>`), or null. */
  project: string | null;
  source: 'harreman' | 'spacetravlr';
  entityKinds: EntityKind[];
  /** Whether `nbhd/<Tier>.json` files exist (older runs predate the neighborhood scores). */
  hasNbhd: boolean;
  /** Whether `beta/<Tier>.json` files exist (only datasets with a SpaceTravLR run have them). */
  hasBeta: boolean;
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
  project: string | null;
  source: 'harreman' | 'spacetravlr';
  entityKinds: EntityKind[];
  tierIds: string[];
  hasNbhd: boolean;
  hasBeta: boolean;
  /**
   * False when the source run is incomplete (e.g. the harreman network JSON exists but no tier
   * tables were written). Such datasets have NO files on disk — the app must list them
   * disabled with `unavailableReason` rather than trying to load them.
   */
  available: boolean;
  unavailableReason?: string;
}
