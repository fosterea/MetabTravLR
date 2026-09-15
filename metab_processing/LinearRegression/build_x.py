"""Build the FULL per-gene design matrix (TF + L-R + L-TF + metab) for a linear-regression
version of SpaceTravLR, attached to a single PROCESSED adata. This is exactly what training
builds via `SpatialCellularProgramsEstimator.init_data()` (`oracles.py::SpaceTravLR.run`) --
we call the real estimator per focus gene and store `train_df` minus the target column. No
new factor math here."""
import sys
import warnings
from pathlib import Path

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


def store_design_matrices(adata, grn, tflinks, focus_genes, *, metabolites=None,
                          radius=300, contact_distance=50, scale_factor=1,
                          tf_ligand_cutoff=0.01, receptor_thresh=0.01,
                          layer='imputed_count', cluster_annot='cell_type_int'):
    """For each focus gene present in `adata.var_names`, build the real
    `SpatialCellularProgramsEstimator` and call `init_data()` -- exactly what training does,
    so `est.train_df` already has all four modulator groups (TF, L-R, L-TF, metab). Store
    `X = train_df.drop(columns=[gene])` as `adata.obsm[f'x_{gene}']` (reindexed to
    `adata.obs_names`) + `adata.uns[f'x_{gene}_cols'] = list(X.columns)`.

    All genes share `adata`, so `init_data`'s diffusion cache (`uns['received_ligands_tfl']`)
    is built once (on the first gene) and reused by the rest -- not cleared between genes. `X`
    is captured right after each `init_data()`, before the per-gene
    `uns['ligand_receptor']`/`['ligand_regulator']`/`['metabolite_interactions']` frames are
    overwritten by the next gene (`train_df` is a fresh owned frame, so this is safe).

    A gene whose `X` has 0 columns (no modulators) is skipped and warned about. Note this is
    looser than training's orphan gate (`oracles.py:551`, which orphans a gene with no TF
    regulators AND no metabolites even if it has L-R/L-TF) -- intentional for this OLS variant:
    we keep a gene with >=1 modulator column from ANY group.
    `adata.uns['x_genes']` lists the genes actually stored. Returns `adata`.
    """
    from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator

    # Never trust a pre-existing received_ligands* cache: a COMMOT setup caches it at
    # radius=350 WITHOUT the metabolite export genes (-> KeyError) and at the wrong scale.
    # Clear it so init_data rebuilds a fresh diffusion at THESE params on the first gene
    # (later genes reuse that fresh cache). Mirrors beta_analysis.compute_metab_x's guard.
    adata.uns.pop('received_ligands', None)
    adata.uns.pop('received_ligands_tfl', None)
    if 'cell_thresholds' in adata.uns:
        warnings.warn(
            "store_design_matrices: cell_thresholds present (COMMOT); L-R diffusion will be "
            "rebuilt filtered by them -- ensure params match the run that produced them.")

    stored_genes = []
    for gene in focus_genes:
        if gene not in adata.var_names:
            warnings.warn(f"store_design_matrices: {gene!r} not in adata.var_names; skipping.")
            continue

        est = SpatialCellularProgramsEstimator(
            adata=adata, target_gene=gene, layer=layer, cluster_annot=cluster_annot,
            radius=radius, contact_distance=contact_distance,
            tf_ligand_cutoff=tf_ligand_cutoff, grn=grn, scale_factor=scale_factor,
            tflinks=tflinks, receptor_thresh=receptor_thresh, metabolites=metabolites,
        )
        est.init_data()

        X = est.train_df.drop(columns=[gene])
        if X.shape[1] == 0:
            warnings.warn(f"store_design_matrices: {gene!r} has no modulators; skipping.")
            continue

        adata.obsm[f'x_{gene}'] = X.reindex(adata.obs_names).to_numpy()
        adata.uns[f'x_{gene}_cols'] = list(X.columns)
        stored_genes.append(gene)

    adata.uns['x_genes'] = stored_genes
    return adata


def build_x_adata(adata, out_path, *, focus_genes, metabolites=None, setup_dir=None,
                  annot='cell_type', cluster_annot='cell_type_int', run_commot=False,
                  radius=300, contact_distance=50, scale_factor=1,
                  tf_ligand_cutoff=0.01, receptor_thresh=0.01, layer='imputed_count'):
    """End-to-end: get processed adata + networks (reuse an existing `setup_dir` or run
    `SpaceShip.setup_`), store all-group design matrices for `focus_genes`, clean up the
    big/gene-specific artifacts, and write ONE adata to `out_path`. Returns the written adata.
    """
    import pandas as pd
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

    store_design_matrices(proc, grn, tflinks, focus_genes, metabolites=metabolites, **params)

    for key in _CLEANUP_OBSM:
        proc.obsm.pop(key, None)
    for key in _CLEANUP_UNS:
        proc.uns.pop(key, None)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    proc.write_h5ad(out_path)
    print(f'[build_x_adata] wrote {out_path}')
    return proc
