"""Build a DEDUPLICATED design-matrix store for a linear-regression version of SpaceTravLR,
attached to a single PROCESSED adata.

Rev 4 (supersedes rev 3's per-gene `obsm['x_{gene}']`): every non-metab factor column
(bare TF gene, `lig$rec`, `lig#tf`) has TARGET-GENE-INDEPENDENT values -- a TF column is
just `imputed_count[TF]`; an L-R/L-TF column is `received(lig) * receptor_or_tf_expr`,
identical wherever it appears. Only WHICH columns a gene uses (its regulators, its L-TF
set, self-exclusion) is gene-specific. So instead of storing a full `cells x modulators`
matrix per gene (duplicating shared columns across genes), we store ONE deduplicated block
of unique factor columns (`obsm['x_factors']`) + a per-gene `{gene: [column names]}` map
(`uns['x_factor_map']`), and reconstruct any gene's matrix on demand via
`get_gene_factors`. Metabolites are kept as their own separate block (`obsm['x_metab']`)
since a `metab@` sum CAN differ for a gene that is itself one of the metabolite's own
transporter genes (self-exclusion), so they are not safely gene-independent to dedup this
way -- see `metab_processing.SpaceTravLR.beta_analysis.compute_metab_x`.
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

from metab_processing.SpaceTravLR.beta_analysis import METAB_PREFIX  # noqa: E402

# Keys mirrored, when present, from betadata/run_params.json onto the matching kwarg.
# NOTE: run_params.json never persists 'receptor_thresh' (SpaceTravLR.__init__'s dump,
# oracles.py:479-495, doesn't write it) -- so receptor_thresh always stays the caller's
# default and has no entry here.
_RUN_PARAM_OVERRIDES = (
    ('radius', 'radius'),
    ('contact_distance', 'contact_distance'),
    ('scale_factor', 'scale_factor'),
    ('tf_ligand_cutoff', 'tf_ligand_cutoff'),
    ('layer', 'layer'),
    ('annot', 'cluster_annot'),
)

# Big / gene-specific artifacts we don't want baked into the written adata.
_CLEANUP_OBSM = ('spatial_maps', 'spatial_features')
_CLEANUP_UNS = (
    'received_ligands', 'received_ligands_tfl',
    'ligand_receptor', 'ligand_regulator', 'metabolite_interactions',
)


def build_factor_block(adata, grn, tflinks, focus_genes, *, radius=300, contact_distance=50,
                       scale_factor=1, tf_ligand_cutoff=0.01, receptor_thresh=0.01,
                       layer='imputed_count', cluster_annot='cell_type_int'):
    """Build the deduplicated NON-METAB factor block for `focus_genes`.

    For each gene, builds the real `SpatialCellularProgramsEstimator` (metab-free) and
    calls `init_data()` -- exactly what training does -- then takes
    `X = est.train_df.drop(columns=[gene])`. Columns are gene-independent by name (a TF
    column, `lig$rec`, or `lig#tf` has the same values wherever it appears), so we merge
    every gene's `X` into ONE union block keyed by column name (first writer wins; later
    genes reuse the value) and record each gene's own column order separately.

    Stores `adata.obsm['x_factors']` (cells x n_unique_cols), `adata.uns['x_factors_cols']`
    (the block's column order), `adata.uns['x_factor_map']` (`{gene: [col names]}`, each
    gene's own order preserved), and `adata.uns['x_genes']` (genes actually stored). A
    focus gene missing from `adata.var_names` is dropped with a warning; a gene whose `X`
    has 0 columns (no modulators) is skipped and warned about. Returns `adata`.
    """
    from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator

    # Never trust a pre-existing received_ligands* cache: a COMMOT setup caches it at a
    # different radius / without our export genes. Clear it so init_data rebuilds a fresh
    # diffusion at THESE params on the first gene (later genes reuse that fresh cache).
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
    stored_genes = []

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
            adata=adata, target_gene=gene, layer=layer, cluster_annot=cluster_annot,
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
        stored_genes.append(gene)

    cols = list(union_cols.keys())
    block = (np.column_stack([union_cols[c] for c in cols]) if cols
             else np.empty((adata.n_obs, 0), dtype=np.float32))

    adata.obsm['x_factors'] = block
    adata.uns['x_factors_cols'] = cols
    adata.uns['x_factor_map'] = x_factor_map
    adata.uns['x_genes'] = stored_genes
    return adata


def get_gene_factors(adata, gene, metabs=None):
    """Reconstruct one gene's design matrix from the shared blocks built by
    `build_factor_block` (+ `add_metabolites`).

    Returns a DataFrame indexed by `adata.obs_names`. Non-metab columns are the gene's
    `uns['x_factor_map'][gene]` names, sliced out of `obsm['x_factors']`. `metabs` selects
    which metabolite columns (from `obsm['x_metab']`) to append:
      - `None` (default): none.
      - `'all'` / `True`: every stored metabolite (`uns['x_metab_modulators']`).
      - a list of names: matched against the stored `metab@<name>` columns by stripping
        the `metab@` prefix (a bare `<name>` or a full `metab@<name>` both work); a
        requested name not found is warned about and skipped. NOTE: a MERGED metabolite
        (D11 -- metabolites with an identical expanded pair-set collapse into one column)
        is named `metab@nameA|nameB`; to select it by list you must pass the full merged
        name (`"nameA|nameB"` or `"metab@nameA|nameB"`), not a single constituent name.

    Raises `KeyError` if `gene` was not stored by `build_factor_block`.
    """
    factor_map = adata.uns.get('x_factor_map', {})
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


def add_metabolites(adata, metabolites, *, radius=300, contact_distance=50, scale_factor=1,
                    layer='imputed_count'):
    """Add/extend the shared metabolite block (`obsm['x_metab']` + `uns['x_metab_modulators']`)
    so more metabolites can be added after the fact, without rebuilding the factor block.

    Reuses `beta_analysis.compute_metab_x` (no hand-rolled diffusion) to compute the new
    `metab@<name>` columns. If no metab block exists yet, stores it directly (matching
    `beta_analysis.metab_x_to_adata`). Otherwise reconstructs the existing block as a
    DataFrame, drops any newly-computed columns whose name already exists, and concatenates
    -- safe because both frames are indexed by this adata's `obs_names`.
    """
    from metab_processing.SpaceTravLR import beta_analysis

    x_new = beta_analysis.compute_metab_x(
        adata, metabolites, radius, contact_distance, scale_factor, layer)

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
                  tf_ligand_cutoff=0.01, receptor_thresh=0.01, layer='imputed_count'):
    """End-to-end: get processed adata + networks (reuse an existing `setup_dir` or run
    `SpaceShip.setup_`), build the deduplicated factor block for `focus_genes`, optionally
    add metabolites, clean up the big/gene-specific artifacts, and write ONE adata to
    `out_path`. Returns the written adata.
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
                  layer=layer, cluster_annot=cluster_annot)
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

    build_factor_block(proc, grn, tflinks, focus_genes, **params)

    if metabolites:
        add_metabolites(proc, metabolites, radius=params['radius'],
                        contact_distance=params['contact_distance'],
                        scale_factor=params['scale_factor'], layer=params['layer'])

    for key in _CLEANUP_OBSM:
        proc.obsm.pop(key, None)
    for key in _CLEANUP_UNS:
        proc.uns.pop(key, None)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    proc.write_h5ad(out_path)
    print(f'[build_x_adata] wrote {out_path}')
    return proc
