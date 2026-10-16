"""Build a DEDUPLICATED design-matrix store for a linear-regression version of SpaceTravLR,
attached to a single PROCESSED adata.

Rev 4 (supersedes rev 3's per-gene `obsm['x_{gene}']`): every non-metab factor column
(bare TF gene, `lig$rec`, `lig#tf`) has TARGET-GENE-INDEPENDENT values -- a TF column is
just `normalized_count[TF]`; an L-R/L-TF column is `received(lig) * receptor_or_tf_expr`,
identical wherever it appears. Only WHICH columns a gene uses (its regulators, its L-TF
set, self-exclusion) is gene-specific. So instead of storing a full `cells x modulators`
matrix per gene (duplicating shared columns across genes), we store ONE deduplicated block
of unique factor columns (`obsm['x_factors']`) + a per-gene `{gene: [column names]}` map
(`uns['x_factor_map']`), and reconstruct any gene's matrix on demand via
`get_gene_factors`. Metabolites are kept as their own separate block (`obsm['x_metab']`)
since a `metab@` sum CAN differ for a gene that is itself one of the metabolite's own
transporter genes (self-exclusion), so they are not safely gene-independent to dedup this
way -- see `metab_processing.SpaceTravLR.beta_analysis.compute_metab_x`.

Rev 6 (2026-09-29): dropped the two-SOURCE ('imputed'/'lognorm') dual-block machinery.
There is now ONE factor block, built entirely on `normalized_count` (the un-imputed
`log1p(raw)` layer -- see `ensure_lognorm_layer`), stored under the plain unsuffixed keys
(`x_factors`, `x_factors_cols`, `x_factor_map`, `x_metab`, `x_metab_modulators`,
`x_genes`). We keep those keys unsuffixed for continuity with the pre-dual-block (rev 4)
scheme, but the VALUES behind them are now built on `normalized_count`, not
`imputed_count`. Since `init_ligands_and_receptors`'s receptor gate already prefers
`normalized_count` when present, building on it needs no special ordering relative to
`imputed_count` (unlike the old two-step imputed-then-lognorm dance).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root (and src/) importable regardless of CWD, matching the other
# metab_processing/SpaceTravLR scripts.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parents[2],
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Mirror of beta_analysis.METAB_PREFIX, kept local so importing build_x for the analysis
# path (get_gene_factors) pulls no heavy data-loader deps (e.g. pyarrow via beta_analysis).
METAB_PREFIX = "metab@"

# The single design-matrix source: the un-imputed log1p(raw) layer. `normalized_count` is
# NOT persisted by `process_adata_` (created then deleted); `build_x_adata` recreates it
# from `raw_count` (see `ensure_lognorm_layer`) before building the factor block.
LAYER = 'normalized_count'

# Keys mirrored, when present, from betadata/run_params.json onto the matching kwarg.
# NOTE: run_params.json never persists 'receptor_thresh' (SpaceTravLR.__init__'s dump,
# oracles.py:479-495, doesn't write it) -- so receptor_thresh always stays the caller's
# default and has no entry here.
_RUN_PARAM_OVERRIDES = (
    ('radius', 'radius'),
    ('contact_distance', 'contact_distance'),
    ('scale_factor', 'scale_factor'),
    ('tf_ligand_cutoff', 'tf_ligand_cutoff'),
    ('annot', 'cluster_annot'),
)

# Big / gene-specific artifacts we don't want baked into the written adata.
_CLEANUP_OBSM = ('spatial_maps', 'spatial_features')
_CLEANUP_UNS = (
    'received_ligands', 'received_ligands_tfl',
    'ligand_receptor', 'ligand_regulator', 'metabolite_interactions',
)


def stored_genes(adata):
    """The genes actually stored (the keys of `x_factor_map`, in insertion order) --
    i.e. the focus genes `build_factor_block` successfully built a non-empty design
    matrix for. Use this as the default gene set."""
    return list(adata.uns.get('x_factor_map', {}).keys())


def ensure_lognorm_layer(adata):
    """Recreate `adata.layers['normalized_count']` (un-imputed log-normalized counts) if
    missing, mirroring `SpaceShip.process_adata_`'s own rule (`spaceship.py`): log1p only
    if `raw.max() > 100`, otherwise use raw as-is. `process_adata_` creates this layer and
    then DELETES it after MAGIC imputation, so it is never in a persisted `_adata.h5ad`
    (only `raw_count` + `imputed_count` are) -- we recreate it here from `raw_count`.

    Mutates `adata` in place and returns it. Raises `ValueError` if both
    `normalized_count` and `raw_count` are missing (nothing to recreate from).
    """
    if 'normalized_count' in adata.layers:
        return adata
    if 'raw_count' not in adata.layers:
        raise ValueError(
            "ensure_lognorm_layer: adata has neither 'normalized_count' nor 'raw_count' "
            "layers; cannot recreate the un-imputed log-normalized layer.")
    raw = adata.layers['raw_count']
    norm = raw.copy()
    if norm.max() > 100:
        norm = np.log1p(norm)
    adata.layers['normalized_count'] = norm
    return adata


def build_factor_block(adata, grn, tflinks, focus_genes, *, radius=300,
                       contact_distance=50, scale_factor=1, tf_ligand_cutoff=0.01,
                       receptor_thresh=0.01, cluster_annot='cell_type_int'):
    """Build the deduplicated NON-METAB factor block for `focus_genes`, on `LAYER`
    (`normalized_count`).

    For each gene, builds the real `SpatialCellularProgramsEstimator` (metab-free) and
    calls `init_data()` -- exactly what training does -- then takes
    `X = est.train_df.drop(columns=[gene])`. Columns are gene-independent by name (a TF
    column, `lig$rec`, or `lig#tf` has the same values wherever it appears), so we merge
    every gene's `X` into ONE union block keyed by column name (first writer wins; later
    genes reuse the value) and record each gene's own column order separately.

    Stores `adata.obsm['x_factors']` (cells x n_unique_cols),
    `adata.uns['x_factors_cols']` (the block's column order),
    `adata.uns['x_factor_map']` (`{gene: [col names]}`, each gene's own order
    preserved), and `adata.uns['x_genes']` (genes actually stored). A focus gene missing
    from `adata.var_names` is dropped with a warning; a gene whose `X` has 0 columns (no
    modulators) is skipped and warned about. Returns `adata`.
    """
    from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator

    # Never trust a pre-existing received_ligands* cache: a COMMOT setup caches it at a
    # different radius / without our export genes, and a prior call would have cached a
    # diffusion at different params. Clear it so init_data rebuilds a fresh diffusion at
    # THESE params on the first gene (later genes reuse that cache).
    adata.uns.pop('received_ligands', None)
    adata.uns.pop('received_ligands_tfl', None)
    if 'cell_thresholds' in adata.uns:
        # COMMOT-masked diffusion is NOT supported here: when cell_thresholds is present,
        # received_ligands is a MASKED frame that differs from the unmasked
        # received_ligands_tfl, so the empty-frame restore below (which repairs the
        # pre-existing bug by copying from _tfl) would silently overwrite correct masked
        # values with wrong unmasked ones. Refuse rather than produce wrong numbers.
        raise ValueError(
            "build_factor_block does not support COMMOT-masked diffusion: adata.uns has "
            "'cell_thresholds'. Rebuild the setup with run_commot=False (the project default).")

    for gene in focus_genes:
        if gene not in adata.var_names:
            warnings.warn(f"build_factor_block: {gene!r} not in adata.var_names; skipping.")

    genes = [g for g in focus_genes if g in adata.var_names]

    x_factor_map = {}
    union_cols = {}
    kept_genes = []

    # Perf fix 1 (memoize get_regulators for this build only): `grn.get_regulators(adata,
    # gene)` is called once per target gene (line ~645 in parallel_estimators.py) AND once
    # per NicheNet ligand column inside init_ligands_and_receptors (~L566) -- the latter are
    # target-gene-INDEPENDENT (same `adata`, same GRN links, same var_names throughout this
    # build), so across ~5000 genes x ~46 ligands this redoes ~46 distinct lookups ~230k
    # times. Wrap the instance method with a per-build memo keyed on `gene` alone (confirmed
    # the only two call sites pass no other varying arg), live only for this call, and
    # restore the `grn` object to its EXACT pre-call state afterward (not merely "patched
    # back to the original method" -- if it had no instance-level `get_regulators` shadow
    # before this call, it must have none after either).
    _had_attr = 'get_regulators' in vars(grn)
    _orig_get_regulators = grn.get_regulators
    _reg_cache = {}

    def _cached_get_regulators(adata_arg, gene, *a, **k):
        # Only cache the no-extra-args shape that both known call sites use; any call with
        # extra positional/keyword args (e.g. a future `alpha=` override) bypasses the
        # cache entirely rather than risk caching a result keyed on `gene` alone that
        # actually depended on those args too.
        if a or k:
            return _orig_get_regulators(adata_arg, gene, *a, **k)
        if gene not in _reg_cache:
            _reg_cache[gene] = list(_orig_get_regulators(adata_arg, gene))
        return list(_reg_cache[gene])  # fresh copy each call -> safe if a caller mutates it

    grn.get_regulators = _cached_get_regulators
    try:
        for i, gene in enumerate(genes):
            # Guard a pre-existing bug (parallel_estimators.py ~991-997): a gene with zero
            # L-R pairs sets uns['received_ligands'] to an EMPTY frame in the shared uns; a
            # later L-R gene then KeyErrors reading it. received_ligands_tfl is never emptied
            # and (COMMOT refused above) always equals received_ligands, so repair ONLY the
            # empty-frame case -- never overwrite an already-valid, non-empty frame.
            if i > 0 and 'received_ligands_tfl' in adata.uns:
                rl = adata.uns.get('received_ligands')
                if rl is None or getattr(rl, 'shape', (0, 0))[1] == 0:
                    adata.uns['received_ligands'] = adata.uns['received_ligands_tfl']

            est = SpatialCellularProgramsEstimator(
                adata=adata, target_gene=gene, layer=LAYER, cluster_annot=cluster_annot,
                radius=radius, contact_distance=contact_distance,
                tf_ligand_cutoff=tf_ligand_cutoff, grn=grn, scale_factor=scale_factor,
                tflinks=tflinks, receptor_thresh=receptor_thresh, metabolites=None,
            )
            est.init_data()

            X = est.train_df.drop(columns=[gene]).reindex(adata.obs_names)
            if X.shape[1] == 0:
                warnings.warn(f"build_factor_block: {gene!r} has no modulators; skipping.")
                continue

            x_factor_map[gene] = list(X.columns)
            for col in X.columns:
                if col not in union_cols:
                    # Perf fix 2 (float32 block): materialize a standalone, OWNED 1-D
                    # float32 copy of this column. `X[col].to_numpy()` on a pandas frame
                    # whose same-dtype columns got consolidated into one block returns a
                    # VIEW into that (n_obs x ~M) block, not an independent array --
                    # `.astype(..., copy=False)` is then a no-op when already float32, so
                    # the stored "column" would pin the ENTIRE per-gene train_df block in
                    # memory for the rest of the build (worse than the old column_stack
                    # path this fix was meant to replace). NOTE: `np.ascontiguousarray`
                    # does NOT reliably fix this -- if the view is already C-contiguous
                    # float32 (the common case for a pandas block row), it returns the
                    # SAME view with no copy (verified). `.astype(float32)` with the
                    # DEFAULT `copy=True` is what numpy guarantees always allocates a new,
                    # data-owning array, regardless of the source's existing dtype/layout.
                    union_cols[col] = X[col].to_numpy().astype(np.float32, copy=True)
            kept_genes.append(gene)
    finally:
        # Restore grn to its EXACT pre-call state: if it had no instance-level
        # get_regulators shadow before, leave none after (don't introduce one via
        # assignment); if it did, put that exact original back.
        if _had_attr:
            grn.get_regulators = _orig_get_regulators
        else:
            del grn.get_regulators

    # Perf fix 2 (float32 + preallocated block, the OOM fix): fill a preallocated float32
    # array column-by-column instead of np.column_stack, which transiently doubles peak
    # memory (the list-of-arrays plus the stacked result).
    cols = list(union_cols.keys())
    if cols:
        block = np.empty((adata.n_obs, len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            block[:, j] = union_cols[c]
    else:
        block = np.empty((adata.n_obs, 0), dtype=np.float32)

    adata.obsm['x_factors'] = block
    adata.uns['x_factors_cols'] = cols
    adata.uns['x_factor_map'] = x_factor_map
    adata.uns['x_genes'] = kept_genes
    return adata


def get_gene_factors(adata, gene, metabs=None, custom=None):
    """Reconstruct one gene's design matrix from the shared blocks built by
    `build_factor_block` (+ `add_metabolites` / `add_neighborhood_scores`).

    Returns a DataFrame indexed by `adata.obs_names`. Non-metab columns are the gene's
    `uns['x_factor_map'][gene]` names, sliced out of `obsm['x_factors']`. `metabs`
    selects which metabolite columns (from `obsm['x_metab']`) to append:
      - `None` (default): none.
      - `'all'` / `True`: every stored metabolite (`uns['x_metab_modulators']`).
      - a list of names: matched against the stored `metab@<name>` columns by stripping
        the `metab@` prefix (a bare `<name>` or a full `metab@<name>` both work); a
        requested name not found is warned about and skipped. NOTE: a MERGED metabolite
        (D11 -- metabolites with an identical expanded pair-set collapse into one column)
        is named `metab@nameA|nameB`; to select it by list you must pass the full merged
        name (`"nameA|nameB"` or `"metab@nameA|nameB"`), not a single constituent name.

    `custom` selects which columns of the general custom-factors block (`obsm['x_custom']`
    / `uns['x_custom_cols']`, e.g. from `add_neighborhood_scores`) to append:
      - `None` (default): none.
      - `True` / `'all'`: every stored custom column.
      - a list of exact column names: a requested name NOT found in `x_custom_cols`
        RAISES `ValueError` (unlike `metabs`, which only warns on a miss) -- selecting a
        custom factor by name is intentionally strict.

    Raises `KeyError` if the factor block was never built (`uns['x_factor_map']` absent)
    or if `gene` was not stored.
    """
    if 'x_factor_map' not in adata.uns:
        raise KeyError(
            "get_gene_factors: 'x_factor_map' not in adata.uns -- the factor block was "
            "never built (call build_factor_block(...) first).")

    factor_map = adata.uns['x_factor_map']
    if gene not in factor_map:
        raise KeyError(f"get_gene_factors: {gene!r} not in adata.uns['x_factor_map']")

    cols = list(factor_map[gene])
    all_cols = list(adata.uns.get('x_factors_cols', []))
    idx = pd.Index(all_cols).get_indexer(cols)
    assert (idx >= 0).all(), "x_factor_map references a column missing from x_factors_cols"
    df = pd.DataFrame(adata.obsm['x_factors'][:, idx], columns=cols, index=adata.obs_names)

    if metabs is not None:
        modulators = list(adata.uns.get('x_metab_modulators', []))
        if metabs is True or metabs == 'all':
            metab_cols = list(modulators)
        else:
            by_bare = {c[len(METAB_PREFIX):]: c for c in modulators if c.startswith(METAB_PREFIX)}
            metab_cols = []
            for name in metabs:
                bare = name[len(METAB_PREFIX):] if name.startswith(METAB_PREFIX) else name
                if bare in by_bare:
                    metab_cols.append(by_bare[bare])
                else:
                    warnings.warn(f"get_gene_factors: metab {name!r} not found in x_metab_modulators.")
        if metab_cols:
            metab_df = pd.DataFrame(adata.obsm['x_metab'], columns=modulators, index=adata.obs_names)
            df = pd.concat([df, metab_df[metab_cols]], axis=1)

    if custom is not None:
        custom_cols_all = list(adata.uns.get('x_custom_cols', []))
        if custom is True or custom == 'all':
            sel_cols = list(custom_cols_all)
        else:
            sel_cols = []
            for name in custom:
                if name not in custom_cols_all:
                    raise ValueError(
                        f"get_gene_factors: custom {name!r} not found in x_custom_cols.")
                sel_cols.append(name)
        if sel_cols:
            custom_df = pd.DataFrame(adata.obsm['x_custom'], columns=custom_cols_all, index=adata.obs_names)
            df = pd.concat([df, custom_df[sel_cols]], axis=1)

    return df


def _factor_genes(factor):
    """Gene symbol(s) a factor column references: the bare TF name, or both sides of a
    `lig$rec` / `lig#tf` pair. A `metab@<name>` column names a metabolite (not a gene),
    so it references none."""
    if factor.startswith(METAB_PREFIX):
        return []
    for sep in ('$', '#'):
        if sep in factor:
            return factor.split(sep)
    return [factor]


def add_gene_signature(adata, name, positive=(), negative=(), factor_mode='union',
                       keep_when_overlap='target', verbose=False):
    """Create a signature pseudo-gene `name` fittable exactly like a real gene, and RETURN
    a NEW adata with it appended (original untouched).

    Score (per cell), on the LAYER ('normalized_count'): sum of the positive genes' values
    minus the negative genes' values. This score becomes the pseudo-gene's expression.

    `name` is appended to var_names and to EVERY layer + X (so anndata stays shape-consistent;
    only LAYER's value matters for fitting -- the score goes into every layer's new column).
    `uns['x_factor_map'][name]` = the union (default) or intersection of the factor-column
    lists of the constituent genes (positive+negative) that are present in x_factor_map; `name`
    is appended to `uns['x_genes']`. So `fit_gene_betas(adata2, genes=[name], ...)` and
    `get_gene_factors(adata2, name)` work with no other change.

    OVERLAP (`keep_when_overlap`): the signature's gene set (positive+negative) can overlap
    the genes its factors reference -- e.g. a TF factor that is itself a signature gene, or a
    `lig$rec` whose receptor is a signature gene -- i.e. a factor that partly predicts its own
    target (leakage). On overlap:
      - `'target'` (default): keep the gene set intact, DROP the overlapping factor(s) -- the
        anti-leakage choice (the signature is defined by its genes; remove self-predicting
        factors).
      - `'factor'`: keep all factors, DROP the overlapping gene(s) from the target/score.
      - `'both'`: keep everything (allow the overlap).
    `metab@` factors name a metabolite, not a gene, so they never count as overlap.
    `verbose=True` prints what is kept and what is dropped.

    Returns a NEW adata; the input's var/X/layers and its `x_factor_map`/`x_genes` are not
    modified. `obsm`/`obsp` arrays are shared by reference with the input (treat the result
    as read-only for those; do not mutate obsm arrays in place).
    """
    import anndata as ad
    import scipy.sparse as sp

    if factor_mode not in ('union', 'intersection'):
        raise ValueError(f"add_gene_signature: invalid factor_mode {factor_mode!r}; "
                         "must be 'union' or 'intersection'.")
    if keep_when_overlap not in ('target', 'factor', 'both'):
        raise ValueError(f"add_gene_signature: invalid keep_when_overlap "
                         f"{keep_when_overlap!r}; must be 'target', 'factor', or 'both'.")
    if name in adata.var_names:
        raise ValueError(f"add_gene_signature: {name!r} already in adata.var_names.")

    def _present(genes, label):
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            warnings.warn(f"add_gene_signature: {label} genes {missing} not in "
                         "adata.var_names; skipping.")
        return present

    # Dedup within each list (preserving first-seen order) BEFORE summing, so a repeated
    # name (e.g. positive=['A','A','B']) doesn't double-count. A gene present in BOTH
    # positive and negative still counts once on each side (nets to ~0), since dedup is
    # applied per-list independently.
    pos_present = list(dict.fromkeys(_present(list(positive), 'positive')))
    neg_present = list(dict.fromkeys(_present(list(negative), 'negative')))

    # Factor set from the constituent genes present in x_factor_map (union/intersection).
    factor_map = adata.uns.get('x_factor_map', {})
    constituents = [g for g in (pos_present + neg_present) if g in factor_map]
    if not constituents:
        warnings.warn(f"add_gene_signature: none of {name!r}'s constituent genes are in "
                     "x_factor_map; the signature will have no modulators.")
        factors = []
    elif factor_mode == 'union':
        factors = []
        seen = set()
        for g in constituents:
            for c in factor_map[g]:
                if c not in seen:
                    seen.add(c)
                    factors.append(c)
    else:  # intersection
        common = set(factor_map[constituents[0]])
        for g in constituents[1:]:
            common &= set(factor_map[g])
        factors = [c for c in factor_map[constituents[0]] if c in common]

    # Resolve overlap between the signature's gene set and the genes its factors reference.
    gene_set = set(pos_present) | set(neg_present)
    overlap = {f: [g for g in _factor_genes(f) if g in gene_set] for f in factors}
    overlap = {f: gs for f, gs in overlap.items() if gs}         # only the overlapping factors
    overlap_genes = sorted({g for gs in overlap.values() for g in gs})
    if overlap:
        if keep_when_overlap == 'target':      # keep the gene set; drop overlapping factors
            factors = [f for f in factors if f not in overlap]
        elif keep_when_overlap == 'factor':    # keep factors; drop overlapping genes from y
            drop = set(overlap_genes)
            pos_present = [g for g in pos_present if g not in drop]
            neg_present = [g for g in neg_present if g not in drop]
        # 'both': keep everything
    if verbose:
        print(f"[add_gene_signature {name!r}] factor_mode={factor_mode!r} "
              f"keep_when_overlap={keep_when_overlap!r}")
        if not overlap:
            print(f"  no overlap: kept {len(factors)} factor(s), "
                  f"{len(pos_present) + len(neg_present)} target gene(s)")
        elif keep_when_overlap == 'target':
            print(f"  dropped {len(overlap)} overlapping factor(s): {sorted(overlap)}")
            print(f"  kept {len(factors)} factor(s); target gene set unchanged ({len(gene_set)})")
        elif keep_when_overlap == 'factor':
            print(f"  dropped {len(overlap_genes)} gene(s) from the target: {overlap_genes}")
            print(f"  kept all {len(factors)} factor(s)")
        else:
            print(f"  kept BOTH (overlap allowed): {dict(sorted(overlap.items()))}")

    # Score (per cell) on LAYER from the (possibly overlap-trimmed) gene set.
    if not pos_present and not neg_present:
        warnings.warn(f"add_gene_signature: no positive/negative genes for {name!r} "
                     f"(after overlap handling); score will be all zeros.")
        score = np.zeros(adata.n_obs, dtype=np.float32)
    else:
        present = pos_present + neg_present
        df = adata[:, present].to_df(LAYER)
        pos_sum = df[pos_present].sum(axis=1).to_numpy() if pos_present else 0.0
        neg_sum = df[neg_present].sum(axis=1).to_numpy() if neg_present else 0.0
        score = pos_sum - neg_sum

    # Keep the score as FLOAT regardless of the target layer's own dtype -- casting to an
    # integer layer's dtype (e.g. raw_count int) would truncate/round the score to 0.
    # np.hstack/scipy.sparse.hstack upcast the layer to float as needed; that's fine here
    # since this analysis adata's non-normalized_count layers aren't re-consumed downstream.
    def _append_col(mat, col):
        col = np.asarray(col, dtype=float).reshape(-1, 1)
        if sp.issparse(mat):
            return sp.hstack([mat, sp.csr_matrix(col)], format=mat.format)
        return np.hstack([mat, col])

    new_X = _append_col(adata.X, score)
    new_layers = {k: _append_col(v, score) for k, v in adata.layers.items()}
    new_var = pd.concat([adata.var, pd.DataFrame(index=[name])])

    # Copy each per-gene factor list so the new adata's x_factor_map shares no mutable
    # list objects with the input's (isolation: mutating new.uns['x_factor_map'][g] must
    # not affect adata.uns['x_factor_map'][g]).
    new_factor_map = {g: list(v) for g, v in factor_map.items()}
    new_factor_map[name] = factors

    new_uns = dict(adata.uns)
    new_uns['x_factor_map'] = new_factor_map
    new_uns['x_genes'] = list(adata.uns.get('x_genes', [])) + [name]

    new_adata = ad.AnnData(
        X=new_X, obs=adata.obs.copy(), var=new_var, uns=new_uns,
        obsm={k: v for k, v in adata.obsm.items()},
        obsp={k: v for k, v in adata.obsp.items()},
        layers=new_layers,
    )
    new_adata.obs_names = adata.obs_names
    new_adata.var_names = list(adata.var_names) + [name]
    return new_adata


def add_metabolites(adata, metabolites, *, radius=300, contact_distance=50, scale_factor=1):
    """Add/extend the shared metabolite block (`obsm['x_metab']` +
    `uns['x_metab_modulators']`) so more metabolites can be added after the fact, without
    rebuilding the factor block.

    Reuses `beta_analysis.compute_metab_x` (no hand-rolled diffusion), on `LAYER`, to
    compute the new `metab@<name>` columns. If no metab block exists yet, stores it
    directly. Otherwise reconstructs the existing block as a DataFrame, drops any
    newly-computed columns whose name already exists, and concatenates -- safe because
    both frames are indexed by this adata's `obs_names`.
    """
    from metab_processing.SpaceTravLR import beta_analysis

    x_new = beta_analysis.compute_metab_x(
        adata, metabolites, radius, contact_distance, scale_factor, LAYER)

    if 'x_metab' not in adata.obsm:
        adata.obsm['x_metab'] = x_new.to_numpy()
        adata.uns['x_metab_modulators'] = list(x_new.columns)
        return adata

    existing_cols = list(adata.uns['x_metab_modulators'])
    existing = pd.DataFrame(adata.obsm['x_metab'], columns=existing_cols, index=adata.obs_names)
    new_cols = [c for c in x_new.columns if c not in existing_cols]
    merged = pd.concat([existing, x_new[new_cols]], axis=1)

    adata.obsm['x_metab'] = merged.to_numpy()
    adata.uns['x_metab_modulators'] = list(merged.columns)
    return adata


def _merge_custom(adata, df):
    """Merge new columns into the shared general-purpose custom-factors block
    (`obsm['x_custom']` + `uns['x_custom_cols']`), mirroring `add_metabolites`' merge
    pattern. IDEMPOTENT BY NAME: a column whose name is already stored is SKIPPED (never
    overwritten) rather than merely deduplicated by value -- callers (e.g.
    `add_neighborhood_scores`) are expected to have already dropped such columns before
    doing any expensive work, but this is also enforced here as the storage contract.

    `df` is indexed by (a subset of) `adata.obs_names` and is reindexed to
    `adata.obs_names` before storing. Mutates `adata` in place and returns it.
    """
    df = df.reindex(adata.obs_names)

    if 'x_custom' not in adata.obsm:
        adata.obsm['x_custom'] = df.to_numpy()
        adata.uns['x_custom_cols'] = list(df.columns)
        return adata

    existing_cols = list(adata.uns['x_custom_cols'])
    new_cols = [c for c in df.columns if c not in existing_cols]
    if not new_cols:
        return adata

    existing = pd.DataFrame(adata.obsm['x_custom'], columns=existing_cols, index=adata.obs_names)
    merged = pd.concat([existing, df[new_cols]], axis=1)

    adata.obsm['x_custom'] = merged.to_numpy()
    adata.uns['x_custom_cols'] = list(merged.columns)
    return adata


def add_neighborhood_scores(adata, gene_sets, *, annot_col=None,
                            add_proximity=False, radius=300, scale_factor=1, eps=1e-3):
    """Add spatial-neighborhood pseudo-ligand scores for gene sets to the shared
    custom-factors block (`obsm['x_custom']` + `uns['x_custom_cols']`, via
    `_merge_custom`), reusing the metabolite path's diffusion primitive
    (`parallel_estimators.compute_radius_weights_fast`) directly rather than
    reimplementing any spatial math.

    Formula (per cell i, per set S):
        nbhd(S)[i] = (scale_factor/N) * sum_k K(i,k) * sum_{g in S} normalized_count[g][k]
    where `K` is the SAME narrow Gaussian kernel used for the metabolite "received
    ligand" diffusion (sigma fixed by `eps`, hard cutoff at `radius`, self-weight 1), and
    N = total cell count. `gene_sets` = `{SET_NAME: [gene, ...]}`; for each set we build
    the per-cell pseudo-ligand `setexpr_S[k] = sum_{g in S present in var_names}
    normalized_count[g][k]` (genes absent from `var_names` are warned about and skipped;
    a set with zero present genes is warned about and skipped), then diffuse it.

    `annot_col` (default None): if given, each set's pseudo-ligand is SPLIT by the
    source cell's annotation value -- one column `f"{ct}-{SET_NAME}"` per cell-type
    value `ct` present in `adata.obs[annot_col].astype(str)`, equal to `setexpr_S` with
    non-`ct` cells zeroed (so after diffusion the column reads "how much of S is
    received from `ct` neighbors").

    `add_proximity` (default False; REQUIRES `annot_col`, else raises `ValueError`): for
    each `ct`, also diffuses an all-ones pseudo-ligand masked to `ct` into a column
    `f"{ct}-proximity"` -- the SAME kernel with expression replaced by 1, i.e. the
    kernel-weighted DENSITY of `ct` around each cell (no expression involved).

    IDEMPOTENT BY NAME (and efficient): any pseudo-ligand column whose name is already
    in `uns['x_custom_cols']` is identified and skipped BEFORE any per-set/per-ct work
    (expression summing, mask building) or diffusion happens, so a repeated call (e.g.
    re-adding the same proximity columns) does no recomputation and never duplicates a
    column. If nothing new remains, returns `adata` unchanged. All remaining new
    pseudo-ligand columns are diffused together in ONE `compute_radius_weights_fast`
    call (one matvec, one KDTree build), then merged via `_merge_custom`.
    """
    from SpaceTravLR.models.parallel_estimators import compute_radius_weights_fast

    if add_proximity and annot_col is None:
        raise ValueError("add_neighborhood_scores: add_proximity=True requires annot_col.")
    # A set named 'proximity' would produce `{ct}-proximity` columns identical in name to the
    # proximity-density columns -- reject rather than silently clobber one with the other.
    if add_proximity and 'proximity' in gene_sets:
        raise ValueError("add_neighborhood_scores: a gene set cannot be named 'proximity' "
                         "when add_proximity=True (its columns would collide).")

    existing_cols = set(adata.uns.get('x_custom_cols', []))

    cts = []
    cell_types = None
    if annot_col is not None:
        cell_types = adata.obs[annot_col].astype(str)
        cts = list(cell_types.unique())

    lig_cols = {}

    for name, genes in gene_sets.items():
        if annot_col is None:
            needed_cts = None if name not in existing_cols else []
        else:
            needed_cts = [ct for ct in cts if f"{ct}-{name}" not in existing_cols]
        if needed_cts is not None and not needed_cts:
            continue  # every column this set would produce already exists

        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            warnings.warn(
                f"add_neighborhood_scores: set {name!r} genes {missing} not in "
                "adata.var_names; skipping those.")
        if not present:
            warnings.warn(
                f"add_neighborhood_scores: set {name!r} has no genes present in "
                "adata.var_names; skipping.")
            continue

        setexpr = adata[:, present].to_df(LAYER).sum(axis=1).to_numpy()

        if annot_col is None:
            lig_cols[name] = setexpr
        else:
            for ct in needed_cts:
                mask = (cell_types == ct).to_numpy()
                lig_cols[f"{ct}-{name}"] = np.where(mask, setexpr, 0.0)

    if add_proximity:
        for ct in cts:
            col = f"{ct}-proximity"
            if col in existing_cols:
                continue
            mask = (cell_types == ct).to_numpy()
            lig_cols[col] = mask.astype(float)

    if not lig_cols:
        return adata

    lig = pd.DataFrame(lig_cols, index=adata.obs_names)
    received = compute_radius_weights_fast(
        adata.obsm['spatial'], lig, radius, scale_factor, eps=eps)

    return _merge_custom(adata, received)


def build_x_adata(adata, out_path, *, focus_genes, metabolites=None, setup_dir=None,
                  annot='cell_type', cluster_annot='cell_type_int', run_commot=False,
                  radius=300, contact_distance=50, scale_factor=1,
                  tf_ligand_cutoff=0.01, receptor_thresh=0.01):
    """End-to-end: get processed adata + networks (reuse an existing `setup_dir` or run
    `SpaceShip.setup_`), build the deduplicated factor block (on `normalized_count`, see
    `LAYER`) over `focus_genes`, optionally add metabolites, clean up the big/gene-specific
    artifacts, and write ONE adata to `out_path`. Returns the written adata.
    """
    import scanpy as sc
    from metab_processing.SpaceTravLR.run_spacetravlr import setup_is_complete

    if setup_dir is None:
        setup_dir = Path(out_path).parent / 'spacetravlr_output'
    setup_dir = Path(setup_dir)

    if setup_is_complete(setup_dir):
        proc = sc.read_h5ad(setup_dir / 'input_data' / '_adata.h5ad')
    else:
        from SpaceTravLR.spaceship import SpaceShip

        a = adata.copy()
        a.obs['cell_type'] = a.obs[annot]
        if 'raw_count' not in a.layers:
            a.layers['raw_count'] = a.X.copy()
        ship = SpaceShip(name='lr_setup', outdir=str(setup_dir), genes=focus_genes)
        ship.setup_(a, overwrite=True, run_commot=run_commot)
        proc = ship.adata

    # run_params.json (if present) is authoritative -- it's what training actually used.
    params = dict(radius=radius, contact_distance=contact_distance, scale_factor=scale_factor,
                  tf_ligand_cutoff=tf_ligand_cutoff, receptor_thresh=receptor_thresh,
                  cluster_annot=cluster_annot)
    run_params_path = setup_dir / 'betadata' / 'run_params.json'
    if run_params_path.is_file():
        import json
        run_params = json.loads(run_params_path.read_text())
        for json_key, kwarg in _RUN_PARAM_OVERRIDES:
            if json_key in run_params:
                params[kwarg] = run_params[json_key]
    print(f'[build_x_adata] effective params: {params}')

    from SpaceTravLR.tools.network import RegulatoryFactory

    grn = RegulatoryFactory(
        colinks_path=str(setup_dir / 'input_data' / 'celloracle_links.pkl'),
        annot=params['cluster_annot'],
    )
    tflinks = pd.read_parquet(setup_dir / 'input_data' / 'tflinks.parquet')

    ensure_lognorm_layer(proc)
    # Perf fix (OOM): the single factor block is built on `normalized_count` only (LAYER);
    # `imputed_count` (the MAGIC layer, ~2.25GB) is unused downstream (fit_gene_betas,
    # get_gene_factors, add_metabolites, compute_metab_x all read `normalized_count`), so
    # drop it before the per-gene loop to free that memory during the build.
    proc.layers.pop('imputed_count', None)
    build_factor_block(proc, grn, tflinks, focus_genes, **params)
    if metabolites:
        add_metabolites(proc, metabolites, radius=params['radius'],
                        contact_distance=params['contact_distance'],
                        scale_factor=params['scale_factor'])

    for key in _CLEANUP_OBSM:
        proc.obsm.pop(key, None)
    for key in _CLEANUP_UNS:
        proc.uns.pop(key, None)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    proc.write_h5ad(out_path)
    print(f'[build_x_adata] wrote {out_path}')
    return proc


def _union_cols(per_sample_cols):
    """Ordered union of several column-name lists (first-seen order across samples)."""
    seen, out = set(), []
    for cols in per_sample_cols:
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def _stack_block(adatas, obsm_key, cols_per_sample, union_cols, fill, verbose, label):
    """Scatter each sample's (cells x its-own-cols) obsm block into one
    (total-cells x union_cols) float32 array, filling columns a sample never built with
    `fill`. Returns None if no sample carries `obsm_key`."""
    present = [obsm_key in a.obsm for a in adatas]
    if not any(present):
        return None
    if not all(present):
        missing = [i for i, p in enumerate(present) if not p]
        warnings.warn(f"merge_x_adatas: {obsm_key!r} missing from {len(missing)} of "
                      f"{len(adatas)} samples; their cells get {fill} for all {label}.")
    col_index = {c: j for j, c in enumerate(union_cols)}
    n_total = sum(a.n_obs for a in adatas)
    block = np.full((n_total, len(union_cols)), fill, dtype=np.float32)
    row = 0
    for a, cols in zip(adatas, cols_per_sample):
        n = a.n_obs
        if obsm_key in a.obsm and cols:
            mat = np.asarray(a.obsm[obsm_key], dtype=np.float32)
            # keep only this sample's columns that survive into union_cols (all of them
            # for 'union'; the shared subset for 'intersection').
            src = [j for j, c in enumerate(cols) if c in col_index]
            dest = [col_index[cols[j]] for j in src]
            if dest:
                block[row:row + n, dest] = mat[:, src]
        row += n
    if verbose:
        widths = [len(c) for c in cols_per_sample]
        print(f"  {label}: {len(union_cols)} union columns "
              f"(per-sample {min(widths)}..{max(widths)}); "
              f"{'identical across samples' if len(set(map(tuple, cols_per_sample))) == 1 else 'differ across samples'}")
    return block


def merge_x_adatas(adatas, keys, *, label='slice', fill=0.0, cols='union', verbose=True):
    """Merge several per-sample factor-store adatas (each as written by `build_x_adata`)
    into ONE adata with the SAME structure, so every downstream function
    (`fit_gene_betas`, `subsample_gene_betas`, `cluster_factors`, `get_gene_factors`,
    `add_gene_signature`, ...) runs on it unchanged.

    Why not plain `anndata.concat`? It DROPS `uns` (so `x_factors_cols` / `x_factor_map` /
    `x_genes` / `x_metab_modulators` vanish and the fitters break) and it cannot stack the
    `x_factors` / `x_metab` blocks when samples built different column sets. This function
    rebuilds those blocks against a reconciled column space and restores the `uns` metadata.

    Column reconciliation (`cols`):
      - `'union'` (default): keep every factor column any sample built; a cell from a sample
        that never built column f gets `fill` (0.0) there. This is lossless for a fit
        restricted to one sample's cells: with `standardize=True` a column that is constant
        within the selected cells (a 0-filled column is constant 0 there) z-scores to all
        zeros and contributes beta 0 -- so subsetting the merged adata back to one sample
        reproduces that sample's standalone fit. For a fit POOLED across samples, a 0-fill
        means "factor absent / no signal" for the samples that lack it; pass
        `onehot_col=<label>` to the fitters to give each sample its own intercept (FWL).
      - `'intersection'`: keep only columns EVERY sample built (no fill; every value real).

    `x_factor_map` is merged per gene as the ordered union (filtered to the kept columns) of
    that gene's factor lists across the samples that have it; `x_genes` / `x_metab_modulators`
    likewise become the kept-column-filtered union. `obs` gains a categorical `<label>` column
    naming each cell's source sample (its `keys` entry), for subsetting and for `onehot_col`.

    Parameters
    ----------
    adatas : list of AnnData   -- the per-sample stores (in the same order as `keys`).
    keys   : list of str       -- a sample name per adata (written to `obs[label]`).
    label  : str               -- the obs column that records the source sample.
    fill   : float             -- value for a (cell, column) a sample never built ('union').
    cols   : 'union'|'intersection'
    verbose: bool              -- print a merge report (cells, column overlap, fill rate).

    Returns a NEW merged AnnData (inputs untouched).
    """
    import anndata as ad

    if len(adatas) != len(keys):
        raise ValueError(f"merge_x_adatas: got {len(adatas)} adatas but {len(keys)} keys.")
    if not adatas:
        raise ValueError("merge_x_adatas: no adatas to merge.")
    if cols not in ('union', 'intersection'):
        raise ValueError(f"merge_x_adatas: invalid cols {cols!r}; 'union' or 'intersection'.")
    for a, k in zip(adatas, keys):
        if 'x_factors_cols' not in a.uns or 'x_factor_map' not in a.uns:
            raise ValueError(f"merge_x_adatas: sample {k!r} has no factor store "
                             "(missing x_factors_cols / x_factor_map).")

    fac_cols_per = [list(a.uns['x_factors_cols']) for a in adatas]
    if cols == 'union':
        union_fac = _union_cols(fac_cols_per)
    else:
        common = set.intersection(*(set(c) for c in fac_cols_per))
        union_fac = [c for c in fac_cols_per[0] if c in common]
    keep_fac = set(union_fac)

    if verbose:
        print(f"[merge_x_adatas] {len(adatas)} samples, "
              f"{sum(a.n_obs for a in adatas)} cells total; cols={cols!r}, fill={fill}")

    # --- core (X / obs / var / layers / spatial) via anndata.concat; our custom blocks are
    #     rebuilt below, so strip them from shells first so concat never sees ragged widths.
    custom_obsm = ('x_factors', 'x_metab')
    shells = []
    for a in adatas:
        shells.append(ad.AnnData(
            X=a.X, obs=a.obs.copy(), var=a.var,
            layers={k: v for k, v in a.layers.items()},
            obsm={k: v for k, v in a.obsm.items() if k not in custom_obsm},
        ))
    merged = ad.concat(shells, keys=list(keys), label=label, join='outer', uns_merge=None)
    merged.obs[label] = merged.obs[label].astype(str).astype('category')
    merged.obs_names_make_unique()

    # --- factor block + metadata.
    block = _stack_block(adatas, 'x_factors', fac_cols_per, union_fac, fill,
                         verbose, 'x_factors')
    merged.obsm['x_factors'] = block
    merged.uns['x_factors_cols'] = union_fac

    merged_map = {}
    for a in adatas:
        for gene, clist in a.uns['x_factor_map'].items():
            kept = [c for c in clist if c in keep_fac]
            if gene not in merged_map:
                merged_map[gene] = list(kept)
            else:
                have = set(merged_map[gene])
                merged_map[gene].extend(c for c in kept if c not in have)
    merged.uns['x_factor_map'] = merged_map
    merged.uns['x_genes'] = list(merged_map.keys())

    # --- metabolite block (same reconciliation), if any sample has one.
    metab_cols_per = [list(a.uns.get('x_metab_modulators', [])) for a in adatas]
    if any(metab_cols_per):
        if cols == 'union':
            union_metab = _union_cols(metab_cols_per)
        else:
            nonempty = [set(c) for c in metab_cols_per if c]
            common_m = set.intersection(*nonempty) if nonempty else set()
            first = next((c for c in metab_cols_per if c), [])
            union_metab = [c for c in first if c in common_m]
        mblock = _stack_block(adatas, 'x_metab', metab_cols_per, union_metab, fill,
                              verbose, 'x_metab')
        if mblock is not None:
            merged.obsm['x_metab'] = mblock
            merged.uns['x_metab_modulators'] = union_metab

    if verbose:
        print(f"  x_genes: {len(merged.uns['x_genes'])} | "
              f"x_metab_modulators: {len(merged.uns.get('x_metab_modulators', []))}")
        if cols == 'union' and block is not None:
            filled = float((block == fill).mean()) * 100
            print(f"  x_factors fill rate: {filled:.1f}% of entries are {fill} "
                  "(0% means every sample built identical columns)")
        print(f"  obs['{label}'] categories: {list(merged.obs[label].cat.categories)}")
    return merged
