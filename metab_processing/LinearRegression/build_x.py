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
                union_cols[col] = X[col].to_numpy()
        kept_genes.append(gene)

    cols = list(union_cols.keys())
    block = (np.column_stack([union_cols[c] for c in cols]) if cols
             else np.empty((adata.n_obs, 0), dtype=np.float32))

    adata.obsm['x_factors'] = block
    adata.uns['x_factors_cols'] = cols
    adata.uns['x_factor_map'] = x_factor_map
    adata.uns['x_genes'] = kept_genes
    return adata


def get_gene_factors(adata, gene, metabs=None):
    """Reconstruct one gene's design matrix from the shared blocks built by
    `build_factor_block` (+ `add_metabolites`).

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

    if metabs is None:
        return df

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

    if not metab_cols:
        return df

    metab_df = pd.DataFrame(adata.obsm['x_metab'], columns=modulators, index=adata.obs_names)
    return pd.concat([df, metab_df[metab_cols]], axis=1)


def add_gene_signature(adata, name, positive=(), negative=(), factor_mode='union'):
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

    Returns a NEW adata; the input's var/X/layers and its `x_factor_map`/`x_genes` are not
    modified. `obsm`/`obsp` arrays are shared by reference with the input (treat the result
    as read-only for those; do not mutate obsm arrays in place).
    """
    import anndata as ad
    import scipy.sparse as sp

    if factor_mode not in ('union', 'intersection'):
        raise ValueError(f"add_gene_signature: invalid factor_mode {factor_mode!r}; "
                         "must be 'union' or 'intersection'.")
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

    if not pos_present and not neg_present:
        warnings.warn(f"add_gene_signature: no positive/negative genes found for "
                     f"{name!r}; score will be all zeros.")
        score = np.zeros(adata.n_obs, dtype=np.float32)
    else:
        present = pos_present + neg_present
        df = adata[:, present].to_df(LAYER)
        pos_sum = df[pos_present].sum(axis=1).to_numpy() if pos_present else 0.0
        neg_sum = df[neg_present].sum(axis=1).to_numpy() if neg_present else 0.0
        score = pos_sum - neg_sum

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
