#!/usr/bin/env python
"""Drop mito genes, keep the top-N HVG (or all genes), then build the LinearRegression
factor store per dataset path via the standalone `linear_factors` builder.

This path is SpaceTravLR-free and CPU-only: it does NOT run `SpaceShip.setup_`, so there
is no MAGIC imputation (the ~50 GB dense `imputed_count` layer the old `build_x_adata`
built and the LR analysis discarded -- the OOM on large whole-transcriptome data). The
GRN is read directly from the shipped base-GRN parquet (all candidate TFs, no regression),
and L-R/L-TF columns are built with the copied Gaussian kernel. Run on a high-memory CPU
node (no GPU needed).

Each `--paths` entry is a dataset directory holding `adata.h5ad`; output is written to
`<path>/LinearRegression/hvg{top_n}_x_adata.h5ad`, or `hvgall_x_adata.h5ad` when
`--top-n all` is given (skips HVG selection entirely; uses every gene left after the
mito drop).
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc

_root = next((p for p in Path(__file__).resolve().parents
              if (p / '.git').exists() or (p / 'setup.py').exists()),
             Path(__file__).resolve().parent)
for _p in (str(_root), str(_root / 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metab_processing.LinearRegression.build_x import ensure_lognorm_layer
from metab_processing.LinearRegression.linear_factors import build_linear_factor_store


def _log(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def _h5ad_is_readable(path):
    """Minimal readability check (file opens, has 'var'/'obs') -- a local copy so hvg_prep
    imports nothing from the SpaceTravLR stack."""
    try:
        with h5py.File(path, 'r') as f:
            return 'var' in f and 'obs' in f
    except Exception:
        return False


def run(path, top_n, annot, tflinks=None):
    path = Path(path)
    tag = 'all' if top_n == 'all' else str(int(top_n))
    out = path / 'LinearRegression' / f'hvg{tag}_x_adata.h5ad'
    if out.exists() and _h5ad_is_readable(out):
        _log(f'{path.name}: {out.name} already present, skipping')
        return

    _log(f'{path.name}: reading {path / "adata.h5ad"}')
    adata = sc.read_h5ad(path / 'adata.h5ad')

    # Drop cells with no `annot` label (e.g. Atera's unlabeled cells are NaN). Otherwise
    # process_adata's label encoding (encode_labels -> sorted(set(labels))) hits a mix of
    # NaN (float) and str and raises "'<' not supported between 'float' and 'str'". Such
    # cells can't be used in a cell-type-conditioned fit anyway.
    if annot in adata.obs:
        n_na = int(adata.obs[annot].isna().sum())
        if n_na:
            _log(f'{path.name}: dropping {n_na} cells with no {annot!r} annotation')
            adata = adata[adata.obs[annot].notna()].copy()

    # HVG selection and the factor build both expect raw integer counts in X -- fail loud if not.
    x = adata.X
    vals = x.data if hasattr(x, 'data') else np.asarray(x).ravel()
    if vals.size and not np.allclose(vals, np.round(vals)):
        raise ValueError(f'{path.name}: adata.X is not raw integer counts.')

    adata = adata[:, ~adata.var_names.str.upper().str.startswith('MT-')].copy()

    if top_n == 'all':
        hvg = list(adata.var_names)
    else:
        n = int(top_n)
        # HVG on a normalized+log copy (skmisc-free), then subset the raw adata, HVG-ranked.
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n, flavor='cell_ranger')
        hvg = tmp.var.loc[tmp.var.highly_variable, 'dispersions_norm'] \
                 .sort_values(ascending=False).index.tolist()
        if len(hvg) != n:
            raise ValueError(f'{path.name}: selected {len(hvg)} HVG != top_n={n}.')
        adata = adata[:, hvg].copy()

    # Build the factor store with the standalone, SpaceTravLR-free builder (no MAGIC
    # imputation). It calls ensure_lognorm_layer internally, which needs raw counts in a
    # layer -- X is raw here (asserted above), so expose it as raw_count by reference.
    if 'raw_count' not in adata.layers and 'normalized_count' not in adata.layers:
        adata.layers['raw_count'] = adata.X
    build_linear_factor_store(adata, hvg, annot_col=annot, tflinks=tflinks)

    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    _log(f'{path.name}: wrote {out} ({len(hvg)} hvg)')


def _run_with_retry(path, top_n, annot, tflinks=None, retries=3, wait=30):
    """Run one dataset, retrying on transient filesystem OSErrors (Savio Lustre:
    errno 5 EIO / errno 108 ESHUTDOWN). Returns True on success, False if it gave up or
    hit a non-retryable error. Never raises."""
    name = Path(path).name
    for attempt in range(1, retries + 1):
        try:
            run(path, top_n, annot, tflinks=tflinks)
            return True
        except OSError as e:
            _log(f'{name}: attempt {attempt}/{retries} OSError: {e}')
            if attempt < retries:
                time.sleep(wait * attempt)      # linear backoff for a transient blip
        except Exception as e:
            _log(f'{name}: non-retryable {type(e).__name__}: {e}')   # e.g. the raw-counts ValueError
            return False
    _log(f'{name}: gave up after {retries} attempts')
    return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths', nargs='+', required=True)
    ap.add_argument('--top-n', required=True,
                     help="int, or the literal string 'all' to skip HVG selection and use "
                          "every gene (after the mito drop).")
    ap.add_argument('--annot', required=True)
    ap.add_argument('--tflinks', default=None,
                     help="optional path to a cached NicheNet ligand_target_<species>.parquet "
                          "(for offline nodes); if omitted it is downloaded from Zenodo.")
    args = ap.parse_args()

    _log(f'top_n={args.top_n}, annot={args.annot}, {len(args.paths)} dataset(s) to run:')
    for p in args.paths:
        _log(f'  {p}')

    failed = [p for p in args.paths
              if not _run_with_retry(p, args.top_n, args.annot, tflinks=args.tflinks)]
    _log(f'done: {len(args.paths) - len(failed)} ok, {len(failed)} failed')
    for p in failed:
        _log(f'  FAILED: {p}')
