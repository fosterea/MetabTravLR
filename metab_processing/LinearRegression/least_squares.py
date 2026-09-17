"""OLS regression over the deduplicated factor store built by `build_x.py`.

Per-gene linear regression (`y = X @ beta + intercept`, no standardization -- raw
coefficients, so magnitude ranking across factors is scale-dependent) via plain
`np.linalg.lstsq` (CPU). Structured so a regularized regressor could be added later
(`_fit_ols` is the one place that would change); only OLS is implemented here.

CAVEAT -- collinear factors: `np.linalg.lstsq` returns the MIN-NORM solution when
columns of X are collinear/near-collinear (common here -- many metabolites share
SLC2A*/ABC* transporter genes). The fitted curve (`X @ beta`, and r2) is still
correct, but individual coefficients are then NOT uniquely determined among the
collinear columns -- `rank_coefficients`' magnitude ranking should be read with that
in mind. A regularized fit (future work) is the fix if unique per-column attribution
is needed.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parents[2],
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metab_processing.LinearRegression.build_x import get_gene_factors, METAB_PREFIX  # noqa: E402

# Factor-group classification by name separator, mirrored locally (from beta_analysis._group)
# so the fitting/analysis path imports no heavy data-loader deps (e.g. pyarrow).
_SEPARATORS = {"@": "metab", "$": "lr", "#": "ltf"}


def _group(modulator):
    """'metab' / 'lr' / 'ltf' by separator, else 'tf' (a bare gene name)."""
    for sep, name in _SEPARATORS.items():
        if sep in modulator:
            return name
    return "tf"


_COLS = ["gene", "factor", "group", "beta", "r2", "n_cells"]


def _select_cells(adata, annot_col, annot_value, cells):
    """Row selection (obs_names) = annotation filter intersected with `cells`."""
    mask = np.ones(adata.n_obs, dtype=bool)
    if annot_col is not None:
        if annot_value is None:
            raise ValueError("_select_cells: annot_value required when annot_col is given")
        mask &= (adata.obs[annot_col].astype(str) == str(annot_value)).to_numpy()
    if cells is not None:
        cells_arr = np.asarray(cells)
        if cells_arr.dtype == bool:
            if len(cells_arr) != adata.n_obs:
                raise ValueError(
                    f"_select_cells: boolean cells mask has length {len(cells_arr)}, "
                    f"expected {adata.n_obs}")
            mask &= cells_arr
        else:
            found = adata.obs_names.isin(cells_arr)
            missing = len(cells_arr) - int(pd.Index(cells_arr).isin(adata.obs_names).sum())
            if missing:
                warnings.warn(f"_select_cells: {missing} of {len(cells_arr)} cell names "
                              f"not found in adata.obs_names; dropped.")
            mask &= found
    return adata.obs_names[mask]


def _fit_ols(X, y):
    """OLS with intercept via `np.linalg.lstsq`. Returns (beta aligned to X's columns,
    r2); the fitted intercept itself is not returned."""
    n, k = X.shape
    design = np.column_stack([X, np.ones(n)])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta, intercept = coefs[:k], coefs[k]
    resid = y - (design @ coefs)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return beta, r2


def fit_gene_betas(adata, genes=None, metabolites='all', *, annot_col=None, annot_value=None,
                   cells=None, layer='imputed_count'):
    """OLS-fit each gene's expression on its factor matrix, over a selected set of cells.

    Returns a tidy DataFrame [gene, factor, group, beta, r2, n_cells] (one row per
    (gene, factor); r2/n_cells are the per-gene fit's, repeated). `metabolites` is
    passed to `get_gene_factors` as `metabs`. Cell selection: `annot_col`/`annot_value`
    filter, intersected with `cells` (a boolean mask or an obs-name array); default =
    all cells. A gene absent from `x_factor_map`, or with fewer than n_factors+1 cells,
    is skipped with a warning.

    CAVEAT: with collinear/correlated factor columns, `_fit_ols` (`np.linalg.lstsq`)
    returns a min-norm solution, so individual `beta` values (and `rank_coefficients`'
    magnitude ranking) are not uniquely determined among the collinear columns -- see
    the module docstring.
    """
    if genes is None:
        genes = list(adata.uns.get('x_genes', []))
    factor_map = adata.uns.get('x_factor_map', {})
    kept_genes = []
    for gene in genes:
        if gene not in factor_map:
            warnings.warn(f"fit_gene_betas: {gene!r} not in x_factor_map; skipping.")
            continue
        kept_genes.append(gene)

    rows = _select_cells(adata, annot_col, annot_value, cells)
    # Densify only the genes we'll actually fit (kept_genes are all in var_names, since
    # x_factor_map is only populated for genes build_factor_block found there).
    expr = adata[:, kept_genes].to_df(layer) if kept_genes else pd.DataFrame(index=adata.obs_names)

    records = []
    for gene in kept_genes:
        X = get_gene_factors(adata, gene, metabs=metabolites).loc[rows]
        if len(rows) < X.shape[1] + 1:
            warnings.warn(f"fit_gene_betas: {gene!r} has too few cells "
                          f"({len(rows)} < {X.shape[1] + 1}); skipping.")
            continue
        y = expr.loc[rows, gene]
        beta, r2 = _fit_ols(X.to_numpy(), y.to_numpy())
        for factor, b in zip(X.columns, beta):
            records.append((gene, factor, _group(factor), b, r2, len(rows)))

    return pd.DataFrame.from_records(records, columns=_COLS)


def subsample_cells(adata, seed, *, frac=None, n=None, annot_col=None, annot_value=None):
    """Sample obs_names without replacement from the eligible pool (all cells, or
    those matching `annot_col == annot_value`). Exactly one of frac/n must be given."""
    if (frac is None) == (n is None):
        raise ValueError("subsample_cells: pass exactly one of frac/n")
    pool = _select_cells(adata, annot_col, annot_value, None)
    if len(pool) == 0:
        raise ValueError("subsample_cells: eligible pool is empty")
    size = round(frac * len(pool)) if frac is not None else n
    if size > len(pool):
        raise ValueError(f"subsample_cells: requested size {size} exceeds pool {len(pool)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=size, replace=False)
    return pool.to_numpy()[idx]


def subsample_metab_betas(adata, gene, metabolite, *, n_subsamples=100, frac=0.8,
                          annot_col=None, annot_value=None, seed0=0, layer='imputed_count'):
    """`metabolite`'s OLS beta for `gene` across `n_subsamples` cell subsamples (seeds
    `seed0..seed0+n_subsamples-1`). Builds X/y ONCE (not per subsample) and only
    row-slices + refits per draw, to avoid re-densifying `adata` ~n_subsamples times.

    NaN when (i) `metab@<metabolite>` is absent from `uns['x_metab_modulators']` --
    global, the same for every gene, not gene-specific -- or (ii) a given subsample
    draws too few cells to fit (`< n_factors + 1`).
    """
    col = METAB_PREFIX + metabolite
    X = get_gene_factors(adata, gene, metabs=[metabolite])
    betas = np.full(n_subsamples, np.nan)
    if col not in X.columns:
        return betas
    ci = list(X.columns).index(col)
    y = adata[:, gene].to_df(layer)[gene]
    for i, seed in enumerate(range(seed0, seed0 + n_subsamples)):
        cells = subsample_cells(adata, seed, frac=frac, annot_col=annot_col,
                                annot_value=annot_value)
        if len(cells) < X.shape[1] + 1:
            continue
        beta, _ = _fit_ols(X.loc[cells].to_numpy(), y.loc[cells].to_numpy())
        betas[i] = beta[ci]
    return betas


def rank_coefficients(betas_df):
    """`betas_df` sorted by |beta| descending (a copy)."""
    order = betas_df['beta'].abs().sort_values(ascending=False).index
    return betas_df.loc[order].copy()


def plot_top_coefficients(betas_df, top=20, ax=None):
    """Horizontal bar chart of the top-`top` |beta| factors. Returns the ax."""
    import matplotlib.pyplot as plt

    ranked = rank_coefficients(betas_df).head(top)
    if ax is None:
        _, ax = plt.subplots()
    labels = ([f"{g}:{f}" for g, f in zip(ranked['gene'], ranked['factor'])]
              if 'gene' in ranked.columns else list(ranked['factor']))
    y_pos = np.arange(len(ranked))[::-1]
    ax.barh(y_pos, ranked['beta'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('beta')
    return ax


def plot_beta_histogram(values, ax=None, bins=30, title=None):
    """Histogram of a 1-D array of betas (NaNs dropped). Returns the ax."""
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(values, bins=bins)
    if title is not None:
        ax.set_title(title)
    return ax
