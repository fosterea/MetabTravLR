#!/usr/bin/env python
"""Drop mito genes, keep the top-N HVG (or all genes), run SpaceTravLR prep
(build_x_adata) per dataset path.

Each `--paths` entry is a dataset directory holding `adata.h5ad`; output is written to
`<path>/LinearRegression/hvg{top_n}_x_adata.h5ad`, or `hvg_all_x_adata.h5ad` when
`--top-n all` is given (skips HVG selection entirely; uses every gene left after the
mito drop).
"""
import argparse
import os
import sys
import time
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
from metab_processing.LinearRegression.build_x import build_x_adata
from metab_processing.SpaceTravLR.run_spacetravlr import _h5ad_is_readable


def _log(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def run(path, top_n, annot):
    path = Path(path)
    tag = 'all' if top_n == 'all' else str(int(top_n))
    out = path / 'LinearRegression' / f'hvg{tag}_x_adata.h5ad'
    if out.exists() and _h5ad_is_readable(out):
        _log(f'{path.name}: {out.name} already present, skipping')
        return

    _log(f'{path.name}: reading {path / "adata.h5ad"}')
    adata = sc.read_h5ad(path / 'adata.h5ad')

    # HVG selection and build_x both expect raw integer counts in X -- fail loud if not.
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

    lr_dir = path / 'LinearRegression'
    build_x_adata(adata, str(out), focus_genes=hvg, annot=annot,
                  setup_dir=str(lr_dir / f'hvg{tag}_setup'))
    _log(f'{path.name}: wrote {out} ({len(hvg)} hvg)')


def _run_with_retry(path, top_n, annot, retries=3, wait=30):
    """Run one dataset, retrying on transient filesystem OSErrors (Savio Lustre:
    errno 5 EIO / errno 108 ESHUTDOWN). Returns True on success, False if it gave up or
    hit a non-retryable error. Never raises."""
    name = Path(path).name
    for attempt in range(1, retries + 1):
        try:
            run(path, top_n, annot)
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
    args = ap.parse_args()

    _log(f'top_n={args.top_n}, annot={args.annot}, {len(args.paths)} dataset(s) to run:')
    for p in args.paths:
        _log(f'  {p}')

    failed = [p for p in args.paths if not _run_with_retry(p, args.top_n, args.annot)]
    _log(f'done: {len(args.paths) - len(failed)} ok, {len(failed)} failed')
    for p in failed:
        _log(f'  FAILED: {p}')
