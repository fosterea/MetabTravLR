#!/usr/bin/env python
"""Locate exactly where imputation hangs, instead of inferring it.

`impute_clusterwise` builds `magic.MAGIC(verbose=0)` -- MAGIC's own progress reporting is
switched off, and its enlighten counter is invisible in a SLURM log, so a stuck run looks
identical to a slow one. This runs the same per-cluster MAGIC calls with the reporting
turned back ON, and installs a `faulthandler` watchdog that dumps the live Python stack on
an interval. If a call hangs, the stack says which line inside MAGIC/graphtools/sklearn it
is sitting on -- which is the one fact none of our timing so far has produced.

    # smallest reproducer first -- this is the one that already misbehaves
    python debug_impute.py --dataset Human_Lung --cells 1000

    # is it the tiny clusters? drop clusters below MAGIC's n_pca (100)
    python debug_impute.py --dataset Human_Lung --cells 1000 --min-cluster-size 150

    # one cluster only, and try the threading/solver knobs
    python debug_impute.py --dataset Human_Lung --cells 5000 --largest-only --n-jobs -1
    python debug_impute.py --dataset Human_Lung --cells 5000 --largest-only --solver approximate

MAGIC 3.0.0 defaults that matter here: knn=5, decay=1, t=3, n_pca=100, solver='exact',
**n_jobs=1** (single-threaded regardless of how many cores the job requested).
"""
import sys
from pathlib import Path

_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parent,
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import faulthandler
import time

import numpy as np
import pandas as pd

from metab_processing.metab_travlr_config import PROJECT_DATA_DIR


def _log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data-dir', default=PROJECT_DATA_DIR)
    parser.add_argument('--cells', type=int, default=1000, help='subsample size')
    parser.add_argument('--annot', default='leiden_scVI_res_0.5')
    parser.add_argument('--min-cluster-size', type=int, default=0,
                        help='skip clusters smaller than this (tests the tiny-cluster theory)')
    parser.add_argument('--largest-only', action='store_true',
                        help='run only the biggest cluster')
    parser.add_argument('--n-jobs', type=int, default=1, help="MAGIC n_jobs (default 1, its own default)")
    parser.add_argument('--solver', default='exact', choices=['exact', 'approximate'])
    parser.add_argument('--n-pca', type=int, default=100)
    parser.add_argument('--genes', default='all_genes',
                        help="what to impute; 'all_genes' is what the pipeline asks for")
    parser.add_argument('--stack-every', type=int, default=120,
                        help='seconds between live stack dumps (0 disables)')
    parser.add_argument('--verbose', type=int, default=2, help='MAGIC verbosity')
    args = parser.parse_args(argv)

    import magic
    import scanpy as sc

    if args.stack_every:
        # Repeating dump: if MAGIC is stuck we get the exact frame, over and over.
        faulthandler.dump_traceback_later(args.stack_every, repeat=True, exit=False)
        _log(f'watchdog: dumping the live stack every {args.stack_every}s')

    path = Path(args.data_dir) / args.dataset / 'adata.h5ad'
    _log(f'reading {path} (backed)')
    adata = sc.read_h5ad(path, backed='r')
    _log(f'{adata.n_obs} cells x {adata.n_vars} genes')
    if args.annot not in adata.obs.columns:
        raise KeyError(f'{args.annot!r} not in obs')

    rng = np.random.default_rng(0)
    m = min(args.cells, adata.n_obs)
    idx = np.sort(rng.choice(adata.n_obs, size=m, replace=False))
    _log(f'subsampling {m} cells')
    sub = adata[idx].to_memory()
    if adata.isbacked:
        adata.file.close()

    X = sub.X
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    df = pd.DataFrame(X, index=sub.obs_names.astype(str), columns=sub.var_names.astype(str))
    labels = pd.Series(np.asarray(sub.obs[args.annot]), index=df.index)

    sizes = labels.value_counts()
    _log(f'{len(sizes)} clusters in the subsample: {list(sizes.values)}')

    todo = [c for c in sizes.index if sizes[c] >= args.min_cluster_size]
    if args.largest_only:
        todo = [sizes.index[0]]
    _log(f'running MAGIC on {len(todo)} cluster(s) '
         f'(min_cluster_size={args.min_cluster_size}, n_jobs={args.n_jobs}, '
         f'solver={args.solver}, n_pca={args.n_pca})')

    results = []
    for cluster in todo:
        block = df.loc[labels == cluster]
        _log(f'--- cluster {cluster!r}: {block.shape[0]} cells x {block.shape[1]} genes ---')
        t0 = time.time()
        try:
            op = magic.MAGIC(verbose=args.verbose, n_jobs=args.n_jobs,
                             solver=args.solver, n_pca=args.n_pca)
            op.fit_transform(block, genes=args.genes)
            dt = time.time() - t0
            _log(f'    OK in {dt:.1f}s')
            results.append((cluster, block.shape[0], round(dt, 1), 'ok'))
        except Exception as exc:
            dt = time.time() - t0
            _log(f'    FAILED after {dt:.1f}s: {type(exc).__name__}: {exc}')
            results.append((cluster, block.shape[0], round(dt, 1), f'{type(exc).__name__}'))

    if args.stack_every:
        faulthandler.cancel_dump_traceback_later()

    print('\ncluster                cells      secs  status', flush=True)
    for cluster, n, dt, status in sorted(results, key=lambda r: -r[1]):
        print(f'{str(cluster):<20} {n:>8} {dt:>9}  {status}', flush=True)
    slow = [r for r in results if r[2] > 30]
    if slow:
        print(f'\n{len(slow)} cluster(s) took over 30s -- sizes {[r[1] for r in slow]}',
              flush=True)
    return results


if __name__ == '__main__':
    main()
