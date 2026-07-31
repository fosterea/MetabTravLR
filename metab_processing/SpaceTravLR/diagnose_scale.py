#!/usr/bin/env python
"""Scale diagnostics for SpaceTravLR setup: what makes a dataset slow, and how it grows.

`setup` spends most of its time in `impute_clusterwise` (oracles.py:84), which runs a
separate `magic.MAGIC().fit_transform()` per cluster of `adata.obs['cell_type']`. MAGIC
builds a kNN graph and diffusion operator over the cells in that cluster, so the cost is
driven by the *largest cluster*, not the total cell count -- and total work goes as the
sum of squares of cluster sizes if it is quadratic.

Two parts, both safe to run while a job is in flight:

  inventory  (cheap, seconds)  Reads each dataset's h5ad in backed mode: shape, X encoding,
                               layers, whether X_umap already exists (if so, setup skips
                               PCA/neighbors/UMAP entirely), and for every categorical obs
                               column the cluster count + size distribution + the quadratic
                               work index sum(m_i^2).
  --bench    (minutes, CPU)    Times the REAL magic.MAGIC().fit_transform() on increasing
                               subsamples of one dataset and fits t ~ m^alpha in log-log.
                               alpha ~1 means cluster size is harmless; alpha ~2 means we
                               must cap cluster size before the million-cell dataset.

    python diagnose_scale.py                                  # inventory, all datasets
    python diagnose_scale.py --dataset Human_Lung --bench     # + scaling benchmark
    python diagnose_scale.py --bench --bench-sizes 2000 4000 8000 16000 32000

Run `--bench` on a compute node (`srun -A <acct> -p savio3 -t 1:00:00 --pty bash`), not a
login node -- it does real work and holds the subsample in memory.

Writes a JSON report (`--out`) alongside the printed summary.
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
import json
import math
import time

import h5py
import numpy as np
import pandas as pd

from metab_processing.metab_travlr_config import PROJECT_DATA_DIR

MAX_CATEGORIES = 300      # obs columns with more distinct values are ids, not clusters


def _fmt(n):
    return f'{n:,}'


def _x_encoding(path):
    """dense vs sparse and dtype, without reading the matrix."""
    try:
        with h5py.File(path, 'r') as f:
            x = f['X']
            enc = x.attrs.get('encoding-type', 'array' if isinstance(x, h5py.Dataset) else 'sparse')
            if isinstance(x, h5py.Group):                       # csr/csc
                dtype = str(x['data'].dtype)
                nnz = int(x['data'].shape[0])
            else:
                dtype, nnz = str(x.dtype), None
            layers = list(f['layers'].keys()) if 'layers' in f else []
            return {'encoding': str(enc), 'dtype': dtype, 'nnz': nnz, 'layers': layers}
    except Exception as exc:                                    # pragma: no cover - env dependent
        return {'error': f'{type(exc).__name__}: {exc}'}


def _cluster_stats(values):
    """Cluster count and the size distribution that drives per-cluster imputation cost."""
    sizes = pd.Series(values).value_counts()
    m = sizes.to_numpy().astype(float)
    n = float(m.sum())
    return {
        'n_clusters': int(len(m)),
        'largest': int(m.max()),
        'median': int(np.median(m)),
        'smallest': int(m.min()),
        # If per-cluster cost ~ m^2, total work ~ sum(m^2). Compared against n^2 (one big
        # cluster) this says how much the existing partition already saves us.
        'sum_sq': float((m ** 2).sum()),
        'sum_sq_over_n_sq': float((m ** 2).sum() / (n ** 2)),
        'top_sizes': [int(v) for v in sizes.head(8)],
    }


def inventory(data_dir, datasets=None):
    """Structural facts per dataset. Backed reads only -- does not load X."""
    import scanpy as sc

    data_dir = Path(data_dir)
    names = datasets or sorted(
        p.name for p in data_dir.iterdir() if (p / 'adata.h5ad').is_file())

    out = {}
    for name in names:
        path = data_dir / name / 'adata.h5ad'
        if not path.is_file():
            out[name] = {'error': f'no adata.h5ad at {path}'}
            continue

        entry = {'path': str(path), 'file_gb': round(path.stat().st_size / 1e9, 2)}
        entry.update(_x_encoding(path))
        try:
            adata = sc.read_h5ad(path, backed='r')
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            out[name] = entry
            continue

        try:
            entry['n_obs'] = int(adata.n_obs)
            entry['n_vars'] = int(adata.n_vars)
            entry['obsm_keys'] = list(adata.obsm.keys())
            # setup skips PCA/neighbors/UMAP entirely when this is already present
            entry['has_X_umap'] = 'X_umap' in adata.obsm
            entry['has_spatial'] = 'spatial' in adata.obsm

            clusters = {}
            for col in adata.obs.columns:
                series = adata.obs[col]
                if series.dtype.name not in ('category', 'object', 'bool'):
                    continue
                if series.nunique(dropna=True) > MAX_CATEGORIES:
                    continue
                clusters[col] = _cluster_stats(series)
            entry['cluster_columns'] = clusters
        finally:
            if adata.isbacked:
                adata.file.close()

        out[name] = entry
    return out


def bench_impute(data_dir, dataset, sizes, annot, seed=0):
    """Time the REAL `impute_clusterwise` on increasing subsamples and fit t ~ m^alpha.

    Unlike `bench_magic`, this includes what the pipeline actually pays around MAGIC:
    `_adata_to_matrix` densifies the layer (`.todense().A.copy()` -> `.transpose()` ->
    `.copy(order='C')` -- three full dense allocations plus a cache-hostile transpose
    copy), the DataFrame construction, and the per-cluster `.loc` slicing. At 278k x 5k
    each of those copies is 5.6 GB, so leaving them out understates the real cost.
    """
    import scanpy as sc

    from SpaceTravLR.oracles import BaseTravLR

    path = Path(data_dir) / dataset / 'adata.h5ad'
    print(f'\nbenchmark (real impute_clusterwise): {path}', flush=True)
    adata = sc.read_h5ad(path, backed='r')
    n_obs, n_vars = adata.n_obs, adata.n_vars
    if annot not in adata.obs.columns:
        raise KeyError(f'{annot!r} not in obs; pass --annot. '
                       f'candidates: {[c for c in adata.obs.columns if "res" in c][:10]}')
    rng = np.random.default_rng(seed)

    results = []
    for m in sizes:
        if m > n_obs:
            print(f'  skip m={_fmt(m)} (only {_fmt(n_obs)} cells)', flush=True)
            continue
        idx = np.sort(rng.choice(n_obs, size=m, replace=False))
        sub = adata[idx].to_memory()
        sub.obs['cell_type'] = sub.obs[annot]
        sub.layers['normalized_count'] = sub.X.copy()
        n_clusters = int(sub.obs['cell_type'].nunique())

        t0 = time.time()
        BaseTravLR.impute_clusterwise(sub, annot='cell_type',
                                      layer='normalized_count', layer_added='imputed_count')
        total_s = time.time() - t0

        dense_gb = m * n_vars * 4 / 1e9
        results.append({'m': int(m), 'total_s': round(total_s, 2),
                        'n_clusters': n_clusters, 'dense_gb': round(dense_gb, 2)})
        print(f'  m={_fmt(m):>9}  impute={total_s:8.1f}s  ({n_clusters} clusters, '
              f'{dense_gb:.1f} GB dense)', flush=True)
        del sub

    if adata.isbacked:
        adata.file.close()

    alpha = None
    if len(results) >= 2:
        xs = np.log([r['m'] for r in results])
        ys = np.log([max(r['total_s'], 1e-6) for r in results])
        alpha = float(np.polyfit(xs, ys, 1)[0])

    return {'dataset': dataset, 'annot': annot, 'mode': 'impute', 'n_obs': int(n_obs),
            'n_vars': int(n_vars), 'points': results, 'alpha': alpha,
            'time_key': 'total_s'}


def bench_magic(data_dir, dataset, sizes, seed=0):
    """Time MAGIC ALONE on increasing subsamples and fit t ~ m^alpha.

    Isolates the algorithm from the marshalling around it -- useful for attributing cost,
    but it is NOT what the pipeline pays. Use `--bench-mode impute` for that.
    """
    import magic
    import scanpy as sc

    path = Path(data_dir) / dataset / 'adata.h5ad'
    print(f'\nbenchmark: {path}', flush=True)
    adata = sc.read_h5ad(path, backed='r')
    n_obs, n_vars = adata.n_obs, adata.n_vars
    rng = np.random.default_rng(seed)

    # Warm up: the first MAGIC call pays import/JIT costs that would otherwise land
    # entirely on the smallest sample and flatten (or invert) the fitted exponent.
    warm = adata[np.sort(rng.choice(n_obs, size=min(300, n_obs), replace=False))].to_memory()
    wx = warm.X
    wx = wx.toarray() if hasattr(wx, 'toarray') else np.asarray(wx)
    magic.MAGIC(verbose=0).fit_transform(
        pd.DataFrame(wx, index=warm.obs_names, columns=warm.var_names), genes='all_genes')
    del warm, wx

    results = []
    for m in sizes:
        if m > n_obs:
            print(f'  skip m={_fmt(m)} (only {_fmt(n_obs)} cells)', flush=True)
            continue
        idx = np.sort(rng.choice(n_obs, size=m, replace=False))

        t0 = time.time()
        sub = adata[idx].to_memory()
        X = sub.X
        X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        df = pd.DataFrame(X, index=sub.obs_names, columns=sub.var_names)
        load_s = time.time() - t0

        t0 = time.time()
        magic.MAGIC(verbose=0).fit_transform(df, genes='all_genes')
        magic_s = time.time() - t0

        gb = df.to_numpy().nbytes / 1e9
        results.append({'m': int(m), 'magic_s': round(magic_s, 2),
                        'load_s': round(load_s, 2), 'input_gb': round(gb, 2)})
        print(f'  m={_fmt(m):>9}  magic={magic_s:8.1f}s  (load {load_s:5.1f}s, '
              f'input {gb:.2f} GB)', flush=True)
        del df, X, sub

    if adata.isbacked:
        adata.file.close()

    alpha = None
    if len(results) >= 2:
        xs = np.log([r['m'] for r in results])
        ys = np.log([max(r['magic_s'], 1e-6) for r in results])
        alpha = float(np.polyfit(xs, ys, 1)[0])

    return {'dataset': dataset, 'mode': 'magic', 'n_obs': int(n_obs), 'n_vars': int(n_vars),
            'points': results, 'alpha': alpha, 'time_key': 'magic_s'}


def _print_inventory(inv):
    print('=' * 78)
    print('INVENTORY')
    print('=' * 78)
    for name, e in inv.items():
        if 'error' in e and 'n_obs' not in e:
            print(f'\n{name}: ERROR {e["error"]}')
            continue
        print(f'\n{name}')
        print(f'  {_fmt(e["n_obs"])} cells x {_fmt(e["n_vars"])} genes   '
              f'file {e["file_gb"]} GB   X={e.get("encoding")} {e.get("dtype")}')
        if e.get('nnz'):
            density = e['nnz'] / (e['n_obs'] * e['n_vars'])
            print(f'  sparse: {_fmt(e["nnz"])} nonzeros ({density:.1%} dense)')
        print(f'  layers={e.get("layers")}  X_umap={e.get("has_X_umap")}  '
              f'spatial={e.get("has_spatial")}')
        if not e.get('has_X_umap'):
            print('  !! no X_umap -> setup will also run PCA + neighbors + UMAP')
        for col, c in sorted(e.get('cluster_columns', {}).items(),
                             key=lambda kv: -kv[1]['largest'])[:8]:
            print(f'    {col:<28} {c["n_clusters"]:>4} clusters  '
                  f'largest={_fmt(c["largest"]):>9}  median={_fmt(c["median"]):>8}  '
                  f'sum(m^2)/n^2={c["sum_sq_over_n_sq"]:.3f}')


def _print_bench(b):
    key = b.get('time_key', 'magic_s')
    print('\n' + '=' * 78)
    print(f'SCALING ({b.get("mode", "magic")})')
    print('=' * 78)
    pts_all = b['points']
    if b['alpha'] is None or len(pts_all) < 2:
        print('not enough points to fit')
        return
    slowest = max(p[key] for p in pts_all)
    if slowest < 5:
        # Sub-second timings are dominated by fixed overhead; a fit on them is noise.
        print(f'  timings too small to fit (slowest {slowest:.2f}s). Re-run with larger '
              f'--bench-sizes, e.g. --bench-sizes 5000 10000 20000 40000')
        return
    a = b['alpha']
    print(f'  fitted exponent alpha = {a:.2f}   (time ~ cells^alpha)')
    if a < 1.3:
        verdict = 'roughly linear -- cluster size is not the problem'
    elif a < 1.7:
        verdict = 'superlinear -- large clusters hurt, worth capping'
    else:
        verdict = 'near-quadratic -- cluster size MUST be capped at scale'
    print(f'  {verdict}')
    pts = b['points']
    if pts:
        ref = pts[-1]
        for target in (100_000, 278_328, 1_157_659):
            est = ref[key] * (target / ref['m']) ** a
            flag = '' if target <= 4 * ref['m'] else '   <- >4x beyond measured, unreliable'
            print(f'  extrapolated to {_fmt(target):>9} cells -> {est / 3600:8.2f} h{flag}')
        print(f'  (measured up to {_fmt(ref["m"])} cells; a power-law fit does not survive '
              f'a regime change such as\n   exact->approximate kNN or the point where the '
              f'dense copies stop fitting in RAM)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=PROJECT_DATA_DIR)
    parser.add_argument('--dataset', action='append', dest='datasets',
                        help='limit to this dataset (repeatable); default all')
    parser.add_argument('--bench', action='store_true',
                        help='also time imputation on subsamples (use a compute node)')
    parser.add_argument('--bench-mode', choices=['impute', 'magic'], default='impute',
                        help="'impute' times the real impute_clusterwise including the dense "
                             "marshalling (default); 'magic' times MAGIC alone")
    parser.add_argument('--bench-dataset', help='dataset to benchmark (default: first given)')
    parser.add_argument('--bench-sizes', type=int, nargs='+',
                        default=[10000, 25000, 50000, 100000],
                        help='subsample sizes for the benchmark')
    parser.add_argument('--annot', default='leiden_scVI_res_0.5',
                        help='obs column used as cell_type for impute-mode clustering')
    parser.add_argument('--out', default='spacetravlr_diagnostics.json')
    args = parser.parse_args(argv)

    report = {'data_dir': str(args.data_dir)}
    inv = inventory(args.data_dir, args.datasets)
    report['inventory'] = inv
    _print_inventory(inv)

    if args.bench:
        target = args.bench_dataset or (args.datasets or sorted(inv))[0]
        if args.bench_mode == 'impute':
            report['bench'] = bench_impute(args.data_dir, target, args.bench_sizes, args.annot)
        else:
            report['bench'] = bench_magic(args.data_dir, target, args.bench_sizes)
        _print_bench(report['bench'])

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f'\nwrote {args.out}  <- send me this file')
    return report


if __name__ == '__main__':
    main()
