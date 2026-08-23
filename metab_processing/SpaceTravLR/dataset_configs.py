"""Per-dataset settings for the SpaceTravLR SLURM runs.

One dict keyed by dataset folder name under ``PROJECT_DATA_DIR``. Anything a dataset
does not set falls back to ``DEFAULTS`` (``slurm`` merges key-by-key, so a dataset can
override just ``time_hours``).

Two things are deliberately *not* per-dataset by default:

- **Target genes.** ``focus_genes`` defaults to ``metab_travlr_config.FOCUS_GENES`` so
  every dataset trains the same targets and the results stay comparable. Override per
  dataset only when you mean to.
- **Metabolite pairs.** Never configured here -- they always come from that dataset's
  own ``easy_download/harreman_outputs/metabolite_selection.yaml``, so each dataset
  trains only the transporter pairs harreman found evidence for.
"""
from __future__ import annotations

import copy
from pathlib import Path

from metab_processing.metab_travlr_config import FOCUS_GENES, PROJECT_DATA_DIR, DATA_DIR as METAB_DATA_DIR

# Sibling of harreman_logs/ -- deliberately OUTSIDE the dataset's spacetravlr_output/,
# which does not exist yet when a fresh setup job starts (SLURM opens the --output file
# before the job runs, so its parent directory must already exist).
LOG_ROOT = f'{METAB_DATA_DIR}/spacetravlr_logs'


DEFAULTS = {
    # --- data ---
    'cell_type_src': 'leiden_scVI_res_0.5',   # adata.obs column copied to 'cell_type' for setup
    'tiers': ['Tier1', 'Tier2', 'Tier3'],     # obs columns to summarize betas over (missing ones are skipped)
    'focus_genes': list(FOCUS_GENES),         # target genes to train -- shared across datasets

    # --- model ---
    'run_commot': False,                      # harreman is our metabolite prior; COMMOT stays off
    'fit_kwargs': {},                         # passed to SpaceShip.fit (max_epochs, learning_rate, radius, ...)

    # --- artifacts ---
    # Modulator group written into adata.obsm['beta_<gene>'] by betas_to_adata.
    # None = all groups (tf + lr + ltf + metab); 'metab' = transporter pairs only.
    # None makes spacetravlr_adata.h5ad much larger -- the run log prints the shapes.
    'beta_group': None,

    # --- slurm (mirrors metab_processing/Harreman/submit_job.ipynb) ---
    # One job does all three stages. Memory on Savio scales with cores (~8 GB each on
    # savio3_gpu), and setup is the memory-hungry part: process_adata_ and CellOracle each
    # copy the whole AnnData, so Human_Lung (278k x 5k, ~5.6 GB per dense copy) peaks near
    # 50 GB and OOM'd an 8-core allocation. 32 cores ~= 256 GB covers that with headroom.
    # Setup is CPU work, but it still needs a GPU node: importing the package pulls in a
    # CUDA-only torch, so a CPU-partition job cannot even start.
    'slurm': {
        'account': 'fc_wagnerlabfca',
        'partition': 'savio3_gpu',
        'qos': 'a40_gpu3_normal',
        'gres': 'gpu:A40:1',
        'cpus_per_task': 32,
        'time_hours': 24,
        'job_name': 'MetabTravLR',
        'python_path': '/global/home/users/fosterangus/.conda/envs/spacetravlr/bin/python',
    },
}


# Per-dataset overrides. `{}` means "all defaults"; any key of DEFAULTS can be set here,
# and `slurm` merges key-by-key so you only name what differs. For example:
#
#     'Human_Lung': {
#         'cell_type_src': 'leiden_scVI_res_1',    # 25 clusters instead of 16
#         'focus_genes': ['CD4', 'CD3E'],          # override the shared gene set
#         'beta_group': 'metab',                   # smaller spacetravlr_adata.h5ad
#         'slurm': {'time_hours': 36},             # everything else stays default
#     },
#
# A typo'd key raises rather than being silently ignored. One-off changes that should not
# be permanent are better passed to submit(): submit('Human_Lung', time_hours=36).
DATASETS = {
    'Primary_Dermal_Melanoma': {},
    'Human_Lung': {},
    'Human_Prostate_Adenocarcinoma': {},
    # Human_Breast has tiny outlier clusters at every resolution (no res_* column is
    # MAGIC-clean, and it has no res_0.25), so switching resolution cannot help. Those
    # few outlier cells are dropped at load instead -- see `_drop_tiny_clusters` in
    # run_spacetravlr.py -- and the dataset keeps the default resolution.
    'Human_Breast': {},
    'Human_Cervical_Cancer': {},
    'FF_Human_Ovarian_Adenocarcinoma': {},
    'FFPE_Human_Ovarian_Cancer': {},
    # UC Xenium samples live under a different data dir ({DATA_DIR}/UC_Xenium), so pass
    # data_dir=f'{DATA_DIR}/UC_Xenium' to submit()/run_spacetravlr for these. Their cells
    # carry a provided `cell_type` annotation (9 types), which is also the single tier we
    # summarize betas over -> metabtravlr_outputs/cell_type/.
    'Sample_1_UC1_inflamed':      {'cell_type_src': 'cell_type', 'tiers': ['cell_type']},
    'Sample_2_UC1_less_inflamed': {'cell_type_src': 'cell_type', 'tiers': ['cell_type']},
}


def get_config(dataset: str) -> dict:
    """Resolved config for `dataset`: DEFAULTS with the dataset's overrides applied."""
    if dataset not in DATASETS:
        raise KeyError(f'unknown dataset {dataset!r}; known: {sorted(DATASETS)}')

    cfg = copy.deepcopy(DEFAULTS)
    override = copy.deepcopy(DATASETS[dataset])
    cfg['slurm'].update(override.pop('slurm', {}))
    unknown = set(override) - set(cfg)
    if unknown:
        raise KeyError(f'{dataset}: unknown config keys {sorted(unknown)}')
    cfg.update(override)
    cfg['dataset'] = dataset
    return cfg


def dataset_paths(dataset: str, data_dir: str = PROJECT_DATA_DIR) -> dict:
    """Every path the run touches. Layout matches quick_start_metab.ipynb."""
    dataset_dir = Path(data_dir) / dataset
    outdir = dataset_dir / 'spacetravlr_output'
    return {
        'dataset_dir': dataset_dir,
        'adata': dataset_dir / 'adata.h5ad',
        'selection_yaml': dataset_dir / 'easy_download' / 'harreman_outputs' / 'metabolite_selection.yaml',
        'outdir': outdir,
        'input_data': outdir / 'input_data',
        'betadata': outdir / 'betadata',
        'metab_outdir': dataset_dir / 'easy_download' / 'metabtravlr_outputs',
        'beta_adata': dataset_dir / 'spacetravlr_adata.h5ad',
        'log_dir': Path(LOG_ROOT) / dataset,
    }
