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

from metab_processing.LinearRegression.build_x import (  # noqa: E402
    get_gene_factors, _source_layer, _SOURCE_SUFFIX, stored_genes,
)

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

def _fit_lasso(X, y, alpha=1.0):
    """
    Lasso (L1 regularized) regression with intercept via scikit-learn.
    Returns (beta aligned to X's columns, r2); the fitted intercept is omitted.
    """
    from sklearn.linear_model import Lasso

    # fit_intercept=True is the default, handling the column of ones automatically
    model = Lasso(alpha=alpha, fit_intercept=True)
    model.fit(X, y)

    # model.coef_ excludes the intercept when fit_intercept=True
    beta = model.coef_

    # model.score returns the coefficient of determination (R^2)
    r2 = model.score(X, y)

    return beta, r2


def _fit(X, y, *, method='OLS', penalty=1.0, standardize=False):
    """Dispatch to `_fit_ols`/`_fit_lasso`, with an optional z-score standardization
    of X's columns first. Returns (beta aligned to X's columns, r2).

    `standardize=True` z-scores each column of X (`(X - mean) / std`, constant
    columns left un-divided) before fitting, so the returned betas are in
    STANDARDIZED units (per-SD of that column) -- no back-transform is applied.
    This is what makes an L1 penalty treat every factor equally regardless of its
    raw scale.
    """
    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        X = (X - mu) / sd
    if method == 'OLS':
        return _fit_ols(X, y)
    elif method == 'l1':
        return _fit_lasso(X, y, alpha=penalty)
    else:
        raise ValueError(f"invalid method: {method!r}")


def fit_gene_betas(adata, genes=None, metabolites='all', *, annot_col=None, annot_value=None,
                   cells=None, source='imputed', method='OLS', penalty=1.0,
                   standardize=False):
    """OLS-fit each gene's expression on its factor matrix, over a selected set of cells.

    `source` (`'imputed'` or `'lognorm'`, see `build_x.SOURCE_LAYERS`) selects BOTH the
    factor block (`get_gene_factors(..., source=source)`) and the y layer
    (`_source_layer(source)`) -- they must agree, so there is no separate `layer` param.
    `'imputed'` reads/writes the ORIGINAL unsuffixed keys (`x_factor_map`, ...);
    `'lognorm'` reads the `_lognorm`-suffixed ones (see `build_x._SOURCE_SUFFIX`).

    Returns a tidy DataFrame [gene, factor, group, beta, r2, n_cells] (one row per
    (gene, factor); r2/n_cells are the per-gene fit's, repeated). `metabolites` is
    passed to `get_gene_factors` as `metabs`. Cell selection: `annot_col`/`annot_value`
    filter, intersected with `cells` (a boolean mask or an obs-name array); default =
    all cells. `genes=None` defaults to THIS source's own stored genes
    (`build_x.stored_genes(adata, source)`) -- NOT a shared gene list -- so a gene built
    for only one of the two sources is never silently dropped from the other's fit. A gene
    absent from that source's factor map, or with fewer than n_factors+1 cells, is skipped
    with a warning.

    `standardize=True` z-scores each factor column before fitting (see `_fit`), so the
    returned betas are per-SD and comparable across factors of different raw scale --
    useful with `method='l1'` so the penalty doesn't favor large-scale factors. Default
    `standardize=False` keeps raw-scale betas (current/original behavior).

    CAVEAT: with collinear/correlated factor columns, `_fit_ols` (`np.linalg.lstsq`)
    returns a min-norm solution, so individual `beta` values (and `rank_coefficients`'
    magnitude ranking) are not uniquely determined among the collinear columns -- see
    the module docstring.
    """
    layer = _source_layer(source)
    sfx = _SOURCE_SUFFIX[source]
    if genes is None:
        genes = stored_genes(adata, source)
    factor_map = adata.uns.get(f'x_factor_map{sfx}', {})
    kept_genes = []
    for gene in genes:
        if gene not in factor_map:
            warnings.warn(f"fit_gene_betas: {gene!r} not in x_factor_map{sfx}; skipping.")
            continue
        kept_genes.append(gene)

    rows = _select_cells(adata, annot_col, annot_value, cells)
    # Densify only the genes we'll actually fit (kept_genes are all in var_names, since
    # x_factor_map{sfx} is only populated for genes build_factor_block found there).
    expr = adata[:, kept_genes].to_df(layer) if kept_genes else pd.DataFrame(index=adata.obs_names)

    records = []
    for gene in kept_genes:
        X = get_gene_factors(adata, gene, metabs=metabolites, source=source).loc[rows]
        if len(rows) < X.shape[1] + 1:
            warnings.warn(f"fit_gene_betas: {gene!r} has too few cells "
                          f"({len(rows)} < {X.shape[1] + 1}); skipping.")
            continue
        y = expr.loc[rows, gene]
        beta, r2 = _fit(X.to_numpy(), y.to_numpy(), method=method, penalty=penalty,
                        standardize=standardize)
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


def subsample_gene_betas(adata, gene, factors=None, *, n_subsamples=100, frac=0.8,
                         annot_col=None, annot_value=None, seed0=0, source='imputed',
                         metabolites='all', method='OLS', penalty=1.0, standardize=False):
    """Beta of each selected factor for `gene` across `n_subsamples` cell subsamples
    (seeds seed0..seed0+n_subsamples-1). Returns a DataFrame of shape
    (n_subsamples, n_selected_factors): row i = subsample i, columns = the selected
    factor names, values = that factor's fitted beta. NOT metabolite-specific: `factors`
    can be ANY modulator (bare TF gene, `lig$rec`, `lig#tf`, or `metab@<name>`).

    `source` (`'imputed'` or `'lognorm'`, see `build_x.SOURCE_LAYERS`) selects BOTH the
    factor block and the y layer (`_source_layer(source)`).

    `factors=None` -> ALL of the gene's factor columns (from get_gene_factors(..., metabs=
    metabolites, source=source)); otherwise a subset (a list of exact factor names; names
    not present are warned about and dropped). A subsample drawing too few cells
    (< n_factors+1) yields a NaN row. Builds X/y ONCE and only row-slices + refits per draw
    (efficiency).
    """
    layer = _source_layer(source)
    X = get_gene_factors(adata, gene, metabs=metabolites, source=source)
    if factors is None:
        selected_factors = list(X.columns)
    else:
        selected_factors = [f for f in factors if f in X.columns]
        missing = [f for f in factors if f not in X.columns]
        if missing:
            warnings.warn(f"subsample_gene_betas: factors {missing} not in {gene!r}'s "
                          f"columns; dropped.")

    selected_idx = [list(X.columns).index(f) for f in selected_factors]
    y = adata[:, gene].to_df(layer)[gene]

    out = np.full((n_subsamples, len(selected_factors)), np.nan)
    for i, seed in enumerate(range(seed0, seed0 + n_subsamples)):
        cells = subsample_cells(adata, seed, frac=frac, annot_col=annot_col,
                                annot_value=annot_value)
        if len(cells) < X.shape[1] + 1:
            continue
        beta, _ = _fit(X.loc[cells].to_numpy(), y.loc[cells].to_numpy(), method=method,
                       penalty=penalty, standardize=standardize)
        out[i] = beta[selected_idx]
    return pd.DataFrame(out, columns=selected_factors)


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
