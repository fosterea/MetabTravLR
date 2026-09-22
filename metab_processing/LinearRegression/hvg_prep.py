#!/usr/bin/env python
"""Drop mito genes, keep the top-N HVG, run SpaceTravLR prep (build_x_adata) per sample."""
import argparse
import os
import sys
from pathlib import Path

# celloracle/genomepy open a diskcache SQLite DB at import time; on Savio's NFS home its
# locking fails ("locking protocol"). Point the cache at node-local /tmp before that import
# (mirrors run_spacetravlr._isolate_cache_dir).
os.environ.setdefault('XDG_CACHE_HOME', f'/tmp/spacetravlr_cache_{os.environ.get("SLURM_JOB_ID", "local")}')
os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)

_root = next((p for p in Path(__file__).resolve().parents
              if (p / '.git').exists() or (p / 'setup.py').exists()),
             Path(__file__).resolve().parent)
for _p in (str(_root), str(_root / 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scanpy as sc
from metab_processing.metab_travlr_config import DATA_DIR
from metab_processing.SpaceTravLR.dataset_configs import dataset_paths
from metab_processing.LinearRegression.build_x import build_x_adata

DATA_ROOT = f'{DATA_DIR}/Alexi_UC_Spliced'
ANNOT = '25_06_11_ICI_5K_Coarse_annotations'


def run(sample, top_n):
    paths = dataset_paths(sample, data_dir=DATA_ROOT)
    adata = sc.read_h5ad(paths['adata'])

    # HVG selection and build_x both expect raw integer counts in X -- fail loud if not.
    x = adata.X
    vals = x.data if hasattr(x, 'data') else np.asarray(x).ravel()
    if vals.size and not np.allclose(vals, np.round(vals)):
        raise ValueError(f'{sample}: adata.X is not raw integer counts.')

    adata = adata[:, ~adata.var_names.str.upper().str.startswith('MT-')].copy()

    # HVG on a normalized+log copy (skmisc-free), then subset the raw adata, HVG-ranked.
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=top_n, flavor='cell_ranger')
    hvg = tmp.var.loc[tmp.var.highly_variable, 'dispersions_norm'] \
             .sort_values(ascending=False).index.tolist()
    if len(hvg) != top_n:
        raise ValueError(f'{sample}: selected {len(hvg)} HVG != top_n={top_n}.')
    adata = adata[:, hvg].copy()

    lr_dir = paths['dataset_dir'] / 'LinearRegression'
    out = lr_dir / f'hvg{top_n}_x_adata.h5ad'
    build_x_adata(adata, str(out), focus_genes=hvg, annot=ANNOT,
                  setup_dir=str(lr_dir / f'hvg{top_n}_setup'))
    print(f'{sample}: wrote {out} ({len(hvg)} hvg)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--top-n', type=int, required=True)
    args = ap.parse_args()
    for s in args.samples:
        run(s, args.top_n)
