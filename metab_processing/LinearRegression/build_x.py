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

Rev 5: two SOURCE blocks in one adata, selectable at fit time. Everything above was
computed on `imputed_count` (MAGIC-smoothed) only; we now also support `normalized_count`
(un-imputed log1p, same scale as `imputed_count` so the receptor-expression filter behaves
consistently). `source='imputed'|'lognorm'` drives which layer feeds the design matrix.

BACKWARD COMPATIBILITY (rev 5.1): `source='imputed'` reuses the ORIGINAL unsuffixed keys
(`x_factors`, `x_factors_cols`, `x_factor_map`, `x_metab`, `x_metab_modulators`,
`x_genes`) -- byte-identical to the pre-dual-block (rev 4) scheme, so an existing
`x_adata.h5ad` written before this change remains readable with no key migration. Only
`source='lognorm'` gets a `_lognorm` suffix on every key. See `_SOURCE_SUFFIX`.
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

# The two design-matrix sources, and the adata layer each is built from. 'imputed' =
# current/original behavior (MAGIC-smoothed). 'lognorm' = un-imputed log1p(raw) -- the
# SAME scale as 'imputed' (both log1p'd), so the receptor-expression filter behaves
# consistently across sources. `normalized_count` is NOT persisted by `process_adata_`
# (created then deleted); `build_x_adata` recreates it from `raw_count` (see
# `ensure_lognorm_layer`) before building the 'lognorm' block.
SOURCE_LAYERS = {'imputed': 'imputed_count', 'lognorm': 'normalized_count'}

# Key suffix per source. 'imputed' is UNSUFFIXED on purpose -- see the module docstring's
# "BACKWARD COMPATIBILITY" note: a pre-dual-block x_adata.h5ad used plain `x_factors` /
# `x_factor_map` / `x_metab` / `x_metab_modulators` / `x_genes`, and those must keep
# working unchanged. Only 'lognorm' is new, so only it gets a suffix.
_SOURCE_SUFFIX = {'imputed': '', 'lognorm': '_lognorm'}

# Keys mirrored, when present, from betadata/run_params.json onto the matching kwarg.
# NOTE: run_params.json never persists 'receptor_thresh' (SpaceTravLR.__init__'s dump,
# oracles.py:479-495, doesn't write it) -- so receptor_thresh always stays the caller's
# default and has no entry here. There is also no 'layer' entry: the source (not a
# run_params.json key) now drives which layer is used.
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


def _validate_source(source):
    if source not in SOURCE_LAYERS:
        raise KeyError(
            f"unknown source {source!r}; expected one of {list(SOURCE_LAYERS)}")


def _source_layer(source):
    """The adata layer `source` reads from. Raises a friendly `KeyError` for an unknown
    source."""
    _validate_source(source)
    return SOURCE_LAYERS[source]


def _suffix(source):
    """The obsm/uns key suffix for `source` (`''` for 'imputed', `'_lognorm'` for
    'lognorm'). Raises a friendly `KeyError` for an unknown source."""
    _validate_source(source)
    return _SOURCE_SUFFIX[source]


def stored_genes(adata, source):
    """The genes actually stored for `source` (the keys of that source's
    `x_factor_map{sfx}`, in insertion order) -- i.e. the focus genes `build_factor_block`
    successfully built a non-empty design matrix for, for THIS source specifically. Use
    this (not a shared/other-source gene list) as a per-source default gene set."""
    sfx = _suffix(source)
    return list(adata.uns.get(f'x_factor_map{sfx}', {}).keys())


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


def build_factor_block(adata, grn, tflinks, focus_genes, *, source='imputed', radius=300,
                       contact_distance=50, scale_factor=1, tf_ligand_cutoff=0.01,
                       receptor_thresh=0.01, cluster_annot='cell_type_int'):
    """Build the deduplicated NON-METAB factor block for `focus_genes`, on `source`'s layer
    (`SOURCE_LAYERS[source]`).

    For each gene, builds the real `SpatialCellularProgramsEstimator` (metab-free) and
    calls `init_data()` -- exactly what training does -- then takes
    `X = est.train_df.drop(columns=[gene])`. Columns are gene-independent by name (a TF
    column, `lig$rec`, or `lig#tf` has the same values wherever it appears), so we merge
    every gene's `X` into ONE union block keyed by column name (first writer wins; later
    genes reuse the value) and record each gene's own column order separately.

    IMPORTANT (caller contract, see the BLOCKER fix in `build_x_adata`):
    `init_ligands_and_receptors` (`parallel_estimators.py`) gates receptor/L-R selection on
    `'normalized_count' if 'normalized_count' in adata.layers else 'imputed_count'` --
    INDEPENDENTLY of the `layer=` this function passes to the estimator (which only
    controls the VALUES). So if `adata.layers['normalized_count']` exists while building
    `source='imputed'`, the imputed block's modulator SET gets wrongly gated on
    `normalized_count` even though its values still come from `imputed_count` -- a
    mismatch vs. real training (which never has a `normalized_count` layer). Callers that
    build both sources on one adata MUST build 'imputed' while `normalized_count` is
    ABSENT (see `build_x_adata`'s two-step order).

    Stores `adata.obsm[f'x_factors{sfx}']` (cells x n_unique_cols),
    `adata.uns[f'x_factors{sfx}_cols']` (the block's column order),
    `adata.uns[f'x_factor_map{sfx}']` (`{gene: [col names]}`, each gene's own order
    preserved), and `adata.uns[f'x_genes{sfx}']` (genes actually stored for THIS source),
    where `sfx = _suffix(source)` -- `''` for 'imputed' (so it reuses the ORIGINAL
    unsuffixed keys, byte-identical to the pre-dual-block scheme) and `'_lognorm'` for
    'lognorm'. A focus gene missing from `adata.var_names` is dropped with a warning; a
    gene whose `X` has 0 columns (no modulators) is skipped and warned about. Returns
    `adata`.
    """
    from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator

    sfx = _suffix(source)
    layer = SOURCE_LAYERS[source]

    # Never trust a pre-existing received_ligands* cache: a COMMOT setup caches it at a
    # different radius / without our export genes, and a prior call for the OTHER source
    # would have cached the wrong layer's diffusion. Clear it so init_data rebuilds a fresh
    # diffusion at THESE params/layer on the first gene (later genes reuse that cache).
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
        kept_genes.append(gene)

    cols = list(union_cols.keys())
    block = (np.column_stack([union_cols[c] for c in cols]) if cols
             else np.empty((adata.n_obs, 0), dtype=np.float32))

    adata.obsm[f'x_factors{sfx}'] = block
    adata.uns[f'x_factors{sfx}_cols'] = cols
    adata.uns[f'x_factor_map{sfx}'] = x_factor_map
    adata.uns[f'x_genes{sfx}'] = kept_genes
    return adata


def get_gene_factors(adata, gene, metabs=None, source='imputed'):
    """Reconstruct one gene's design matrix from `source`'s shared blocks built by
    `build_factor_block` (+ `add_metabolites`).

    Returns a DataFrame indexed by `adata.obs_names`. Non-metab columns are the gene's
    `uns[f'x_factor_map{sfx}'][gene]` names, sliced out of `obsm[f'x_factors{sfx}']`
    (`sfx = _suffix(source)`: `''` for 'imputed' -- the ORIGINAL unsuffixed keys -- and
    `'_lognorm'` for 'lognorm'). `metabs` selects which metabolite columns (from
    `obsm[f'x_metab{sfx}']`) to append:
      - `None` (default): none.
      - `'all'` / `True`: every stored metabolite (`uns[f'x_metab{sfx}_modulators']`).
      - a list of names: matched against the stored `metab@<name>` columns by stripping
        the `metab@` prefix (a bare `<name>` or a full `metab@<name>` both work); a
        requested name not found is warned about and skipped. NOTE: a MERGED metabolite
        (D11 -- metabolites with an identical expanded pair-set collapse into one column)
        is named `metab@nameA|nameB`; to select it by list you must pass the full merged
        name (`"nameA|nameB"` or `"metab@nameA|nameB"`), not a single constituent name.

    Raises `KeyError` if `source` is unknown, if `source`'s factor block was never built
    (`uns[f'x_factor_map{sfx}']` absent), or if `gene` was not stored for `source`.
    """
    sfx = _suffix(source)
    factor_map_key = f'x_factor_map{sfx}'
    if factor_map_key not in adata.uns:
        raise KeyError(
            f"get_gene_factors: {factor_map_key!r} not in adata.uns -- source {source!r} "
            f"was never built (call build_factor_block(..., source={source!r}) first).")

    factor_map = adata.uns[factor_map_key]
    if gene not in factor_map:
        raise KeyError(f"get_gene_factors: {gene!r} not in adata.uns[{factor_map_key!r}]")

    cols = list(factor_map[gene])
    all_cols = list(adata.uns.get(f'x_factors{sfx}_cols', []))
    idx = pd.Index(all_cols).get_indexer(cols)
    assert (idx >= 0).all(), "x_factor_map references a column missing from x_factors_cols"
    df = pd.DataFrame(adata.obsm[f'x_factors{sfx}'][:, idx], columns=cols, index=adata.obs_names)

    if metabs is None:
        return df

    modulators = list(adata.uns.get(f'x_metab{sfx}_modulators', []))
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

    metab_df = pd.DataFrame(adata.obsm[f'x_metab{sfx}'], columns=modulators, index=adata.obs_names)
    return pd.concat([df, metab_df[metab_cols]], axis=1)


def add_metabolites(adata, metabolites, *, source='imputed', radius=300, contact_distance=50,
                    scale_factor=1):
    """Add/extend `source`'s shared metabolite block (`obsm[f'x_metab{sfx}']` +
    `uns[f'x_metab{sfx}_modulators']`, `sfx = _suffix(source)`) so more metabolites can be
    added after the fact, without rebuilding the factor block.

    Reuses `beta_analysis.compute_metab_x` (no hand-rolled diffusion), on `source`'s layer
    (`SOURCE_LAYERS[source]`), to compute the new `metab@<name>` columns. If no metab block
    exists yet for `source`, stores it directly (matching `beta_analysis.metab_x_to_adata`
    for 'imputed', whose keys are unsuffixed). Otherwise reconstructs the existing block as
    a DataFrame, drops any newly-computed columns whose name already exists, and
    concatenates -- safe because both frames are indexed by this adata's `obs_names`.
    """
    from metab_processing.SpaceTravLR import beta_analysis

    sfx = _suffix(source)
    layer = SOURCE_LAYERS[source]
    obsm_key = f'x_metab{sfx}'
    uns_key = f'x_metab{sfx}_modulators'

    x_new = beta_analysis.compute_metab_x(
        adata, metabolites, radius, contact_distance, scale_factor, layer)

    if obsm_key not in adata.obsm:
        adata.obsm[obsm_key] = x_new.to_numpy()
        adata.uns[uns_key] = list(x_new.columns)
        return adata

    existing_cols = list(adata.uns[uns_key])
    existing = pd.DataFrame(adata.obsm[obsm_key], columns=existing_cols, index=adata.obs_names)
    new_cols = [c for c in x_new.columns if c not in existing_cols]
    merged = pd.concat([existing, x_new[new_cols]], axis=1)

    adata.obsm[obsm_key] = merged.to_numpy()
    adata.uns[uns_key] = list(merged.columns)
    return adata


def build_x_adata(adata, out_path, *, focus_genes, metabolites=None, setup_dir=None,
                  annot='cell_type', cluster_annot='cell_type_int', run_commot=False,
                  radius=300, contact_distance=50, scale_factor=1,
                  tf_ligand_cutoff=0.01, receptor_thresh=0.01):
    """End-to-end: get processed adata + networks (reuse an existing `setup_dir` or run
    `SpaceShip.setup_`), build the deduplicated factor block for BOTH sources
    (`'imputed'` and `'lognorm'`, see `SOURCE_LAYERS`) over `focus_genes`, optionally add
    metabolites for both sources, clean up the big/gene-specific artifacts, and write ONE
    adata (carrying `imputed_count`, `normalized_count`, `raw_count` layers + both
    sources' x blocks) to `out_path`. Returns the written adata.

    BLOCKER fix (build order): the 'imputed' block is built FIRST, while
    `adata.layers['normalized_count']` is ABSENT (popped if present -- e.g. on a reused
    setup dir/adata from a prior run of this function). This matters because
    `init_ligands_and_receptors` gates receptor/L-R selection on
    `'normalized_count' if present else 'imputed_count'`, independent of the `layer=` the
    estimator is given -- so if `normalized_count` existed while building 'imputed', the
    modulator SET would wrongly be gated on `normalized_count` (values still from
    `imputed_count`), diverging from real training (which never has `normalized_count`).
    Only AFTER the imputed block is built do we call `ensure_lognorm_layer` to (re)create
    `normalized_count` and build the 'lognorm' block. `normalized_count` is kept in the
    final written adata (the 'lognorm' fits need it as their y layer).
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

    # Step 1: 'imputed' block, with normalized_count ABSENT so init_ligands_and_receptors'
    # receptor gate falls back to imputed_count (matching real training) -- see the BLOCKER
    # fix in the docstring. Idempotent on reuse: pop any normalized_count a prior run of
    # this function (or a reused setup dir) may have left on the adata.
    proc.layers.pop('normalized_count', None)
    build_factor_block(proc, grn, tflinks, focus_genes, source='imputed', **params)
    if metabolites:
        add_metabolites(proc, metabolites, source='imputed', radius=params['radius'],
                        contact_distance=params['contact_distance'],
                        scale_factor=params['scale_factor'])

    # Step 2: recreate normalized_count from raw_count, THEN build 'lognorm'.
    ensure_lognorm_layer(proc)
    build_factor_block(proc, grn, tflinks, focus_genes, source='lognorm', **params)
    if metabolites:
        add_metabolites(proc, metabolites, source='lognorm', radius=params['radius'],
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
