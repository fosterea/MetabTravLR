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

(Module docstring predates later additions kept here for history: `method='l1'/'elastic'`
regularized fits, `standardize=`, correlation-based factor-group reduction
(`cluster_factors`/`groups=`), and an unpenalized one-hot covariate via
Frisch-Waugh-Lovell (`onehot_col=`) -- see `_fit`'s docstring for the current contract.)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parents[2],
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metab_processing.LinearRegression.build_x import (  # noqa: E402
    get_gene_factors, stored_genes, LAYER,
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


def _group_label(name):
    """Like `_group`, but a reduced cluster column (`cluster_<id>`, from
    `_reduce_groups`) is labeled 'cluster' rather than mis-tagged by `_group` (which
    would read it as a bare TF gene)."""
    return "cluster" if name.startswith("cluster_") else _group(name)


_COLS = ["gene", "factor", "group", "beta", "r", "r2", "model_r2", "n_cells"]


def cluster_factors(adata, gene, *, metabs=None, annot_col=None, annot_value=None,
                    threshold=0.5, method='average'):
    """Hierarchically cluster `gene`'s factor columns by correlation distance.

    Builds the factor matrix with `get_gene_factors(adata, gene, metabs=metabs)`,
    restricted to the selected cells (`annot_col`/`annot_value` -- a value or a list;
    default all cells, same selection as the fit functions), then: correlation matrix
    -> `1 - corr` distance -> average/etc. linkage -> flat clusters cut at `threshold`
    (distance criterion). Returns `{group_id: [factor names]}` suitable for the
    `groups=` argument of `fit_gene_betas`/`subsample_gene_betas`.

    `metabs`: None (default, no metabolites) / 'all' / a list -> passed through to
    `get_gene_factors`. Correlation is scale-invariant, so no standardization is needed
    here. A zero-variance (constant over the selected cells) column has no correlation
    structure (`np.corrcoef` would emit NaNs, which `squareform`/`linkage` reject) and
    is dropped before clustering -- it isn't grouped, so it stays an ungrouped
    individual factor at `_reduce_groups` time. With fewer than 2 non-constant columns
    there is nothing to cluster: each remaining column is returned as its own group.
    """
    rows = _select_cells(adata, annot_col, annot_value, None)
    X = get_gene_factors(adata, gene, metabs=metabs).loc[rows]
    X = X.loc[:, X.std() > 0]
    if X.shape[1] < 2:
        return {i + 1: [c] for i, c in enumerate(X.columns)}
    corr = np.corrcoef(X.values.T)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method)
    cluster_labels = fcluster(link, t=threshold, criterion='distance')
    groups = {}
    for name, label in zip(X.columns, cluster_labels):
        groups.setdefault(int(label), []).append(name)
    return groups


def _reduce_groups(X, groups, standardize):
    """Collapse each group's present-in-`X` factors into one summed `cluster_{id}`
    column; factors not in any group are left untouched. If `standardize`, each
    constituent is z-scored (`(col - mean) / (std or 1)`) before summing, so the
    group's columns are balanced before being combined.

    Column order: ungrouped columns (in `X`'s original order), then one `cluster_{id}`
    per non-empty group, ordered by sorted group id.
    """
    grouped_cols = {c for members in groups.values() for c in members if c in X.columns}
    ungrouped = [c for c in X.columns if c not in grouped_cols]

    data = {c: X[c] for c in ungrouped}
    for gid in sorted(groups):
        members = [c for c in groups[gid] if c in X.columns]
        if not members:
            continue
        if standardize:
            parts = []
            for c in members:
                col = X[c]
                sd = col.std()
                parts.append((col - col.mean()) / (sd if sd else 1))
            summed = sum(parts)
        else:
            summed = X[members].sum(axis=1)
        data[f"cluster_{gid}"] = summed

    return pd.DataFrame(data, index=X.index)


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


def _fit_elastic(X, y, alpha=1.0, l1_ratio=0.5):
    """
    Elastic-net regression with intercept via scikit-learn.
    Returns (beta aligned to X's columns, r2); the fitted intercept is omitted.
    """
    from sklearn.linear_model import ElasticNet

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)
    model.fit(X, y)
    beta = model.coef_
    r2 = model.score(X, y)
    return beta, r2


def _fit(X, y, *, method='OLS', penalty=1.0, l1_ratio=0.5, standardize=False, D=None):
    """Dispatch to `_fit_ols`/`_fit_lasso`/`_fit_elastic`, with an optional z-score
    standardization of X's columns first, and an optional Frisch-Waugh-Lovell (FWL)
    path when `D` is given. Returns (beta, bD, r2).

    `standardize=True` z-scores each column of X (`(X - mean) / std`, constant
    columns left un-divided) before fitting -- X only; `D` is never standardized.
    This is what makes an L1/elastic-net penalty treat every factor equally
    regardless of its raw scale.

    `D=None` (default): `beta` = the fit aligned to X's columns, `bD=None`, `r2` is
    that fit's R^2 -- i.e. exactly the previous (pre-FWL) behavior.

    `D` given (an (n_cells x k) unpenalized covariate design, e.g. one-hot dummy
    columns): D's coefficients are made UNPENALIZED via FWL -- an intercept+D is
    projected out of X and y, the penalized model is fit on the residuals (no
    intercept), and D's (unpenalized) coefficients are recovered from
    `y - X @ beta`. Returns `(beta, bD, r2)` where `bD` is aligned to D's columns
    (intercept dropped) and `r2` is computed from the FULL prediction
    (`X @ beta + Dfull @ bDfull`).
    """
    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        X = (X - mu) / sd

    if D is None:
        if method == 'OLS':
            beta, r2 = _fit_ols(X, y)
        elif method == 'l1':
            beta, r2 = _fit_lasso(X, y, alpha=penalty)
        elif method == 'elastic':
            beta, r2 = _fit_elastic(X, y, alpha=penalty, l1_ratio=l1_ratio)
        else:
            raise ValueError(f"invalid method: {method!r}")
        return beta, None, r2

    n = X.shape[0]
    Dfull = np.column_stack([np.ones(n), D])
    Dpinv = np.linalg.pinv(Dfull)

    def _proj(M):
        return M - Dfull @ (Dpinv @ M)

    Xr, yr = _proj(X), _proj(y)
    if method == 'OLS':
        bX, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    elif method == 'l1':
        from sklearn.linear_model import Lasso
        bX = Lasso(alpha=penalty, fit_intercept=False).fit(Xr, yr).coef_
    elif method == 'elastic':
        from sklearn.linear_model import ElasticNet
        bX = ElasticNet(alpha=penalty, l1_ratio=l1_ratio, fit_intercept=False).fit(Xr, yr).coef_
    else:
        raise ValueError(f"invalid method: {method!r}")

    bDfull = Dpinv @ (y - X @ bX)
    bD = bDfull[1:]
    yhat = X @ bX + Dfull @ bDfull
    resid = y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return bX, bD, r2


def fit_gene_betas(adata, genes=None, metabolites='all', *, annot_col=None, annot_value=None,
                   cells=None, method='OLS', penalty=1.0, l1_ratio=0.5, standardize=False,
                   groups=None, onehot_col=None, just_skip_samples_without_all_labels=False):
    """OLS-fit each gene's expression on its factor matrix, over a selected set of cells.

    Both the factor block (`get_gene_factors`) and the y layer (`LAYER` =
    `'normalized_count'`) come from the single build_x factor store.

    Returns a tidy DataFrame [gene, factor, group, beta, r, r2, model_r2, n_cells] (one
    row per (gene, factor)). `r` = Pearson correlation between the target y and that
    factor's x, `r2` = r**2 (both per-factor); `model_r2` = the whole fit's R² and
    `n_cells` are per-gene, repeated across the gene's rows. `metabolites` is
    passed to `get_gene_factors` as `metabs`. Cell selection: `annot_col`/`annot_value`
    filter, intersected with `cells` (a boolean mask or an obs-name array); default =
    all cells. `genes=None` defaults to `build_x.stored_genes(adata)`. A gene absent from
    the factor map, or with fewer than n_factors+1 cells, is skipped with a warning.

    `standardize=True` z-scores each factor column before fitting (see `_fit`), so the
    returned betas are per-SD and comparable across factors of different raw scale --
    useful with `method='l1'`/`'elastic'` so the penalty doesn't favor large-scale
    factors. Default `standardize=False` keeps raw-scale betas (original behavior).

    `method='elastic'` fits an sklearn `ElasticNet(alpha=penalty, l1_ratio=l1_ratio)`
    (`l1_ratio` default 0.5; unused by `'OLS'`/`'l1'`).

    `groups` (default None): an optional `{group_id: [factor names]}` map (e.g. from
    `cluster_factors`). When given, each gene's factor matrix is reduced via
    `_reduce_groups(X, groups, standardize)` BEFORE fitting -- each group's
    present-in-X factors collapse into one summed `cluster_{id}` column; a reduced
    column's `group` in the output is `'cluster'` (not a TF/LR/LTF misclassification).

    `onehot_col` (default None): an `adata.obs` column to include as an UNPENALIZED
    one-hot covariate via Frisch-Waugh-Lovell (see `_fit`). Labels are
    `sorted(adata.obs[onehot_col].loc[rows].unique())`; the first (reference) label is
    zeroed and the rest get their fitted dummy coefficient. One extra output row per
    label is appended per gene: factor name `f'{onehot_col}[{label}]'`, group
    `'onehot'`, `r` = Pearson(dummy column, y) (computed the same way for every label,
    including the reference). `just_skip_samples_without_all_labels` is accepted for
    signature parity with `subsample_gene_betas` but unused here -- this single-fit
    path always derives labels from its own fitted cells, so every label is present by
    construction.

    CAVEAT: with collinear/correlated factor columns, `_fit_ols` (`np.linalg.lstsq`)
    returns a min-norm solution, so individual `beta` values (and `rank_coefficients`'
    magnitude ranking) are not uniquely determined among the collinear columns -- see
    the module docstring.
    """
    if genes is None:
        genes = stored_genes(adata)
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
    expr = adata[:, kept_genes].to_df(LAYER) if kept_genes else pd.DataFrame(index=adata.obs_names)

    # Onehot covariate: fix labels/reference ONCE over `rows` (shared across every gene
    # in this call, since `rows` doesn't vary by gene), built in the same row order as
    # X/y so the dummy arrays align.
    labels, dummy_labels, dummy_arrays, D_full = [], [], {}, None
    if onehot_col is not None:
        obs_vals = adata.obs[onehot_col].astype(str).loc[rows]
        labels = sorted(obs_vals.unique())
        dummy_labels = labels[1:]
        dummy_arrays = {lbl: (obs_vals == lbl).to_numpy(dtype=float) for lbl in labels}
        D_full = (np.column_stack([dummy_arrays[lbl] for lbl in dummy_labels])
                  if dummy_labels else np.zeros((len(rows), 0)))
    # DOF guard must also count the unpenalized onehot dummy columns (+ their
    # intercept), else an underdetermined fit with onehot_col set can pass silently.
    n_dummy_cols = len(dummy_labels)

    records = []
    for gene in kept_genes:
        X = get_gene_factors(adata, gene, metabs=metabolites).loc[rows]
        if groups is not None:
            X = _reduce_groups(X, groups, standardize)
        min_cells = X.shape[1] + n_dummy_cols + 1
        if len(rows) < min_cells:
            warnings.warn(f"fit_gene_betas: {gene!r} has too few cells "
                          f"({len(rows)} < {min_cells}); skipping.")
            continue
        y = expr.loc[rows, gene]
        Xv, yv = X.to_numpy(), y.to_numpy()
        beta, bD, model_r2 = _fit(Xv, yv, method=method, penalty=penalty, l1_ratio=l1_ratio,
                                  standardize=standardize, D=D_full)
        # Per-factor Pearson r between the target y and that factor's x (scale-invariant, so
        # unaffected by `standardize`); r2 = r**2. 0 for a constant column / constant y.
        Xc, yc = Xv - Xv.mean(axis=0), yv - yv.mean()
        denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.where(denom > 0, (Xc * yc[:, None]).sum(axis=0) / denom, 0.0)
        for factor, b, rj in zip(X.columns, beta, r):
            records.append((gene, factor, _group_label(factor), b, rj, rj ** 2, model_r2, len(rows)))

        if bD is not None:
            for i, lbl in enumerate(labels):
                dummy = dummy_arrays[lbl]
                dc = dummy - dummy.mean()
                denom_o = np.sqrt((dc ** 2).sum() * (yc ** 2).sum())
                r_lbl = float((dc * yc).sum() / denom_o) if denom_o > 0 else 0.0
                b_lbl = 0.0 if i == 0 else float(bD[i - 1])
                records.append((gene, f'{onehot_col}[{lbl}]', 'onehot', b_lbl, r_lbl,
                                 r_lbl ** 2, model_r2, len(rows)))

    return pd.DataFrame.from_records(records, columns=_COLS)


def resample_cells(adata, seed, *, annot_col=None, annot_value=None):
    """Resample the eligible cells (all cells, or those matching `annot_col ==
    annot_value`) WITH replacement to the same size as the pool -- a uniform draw with
    replacement over the pool, so the result is the same length as the pool but with
    duplicates expected. Raises `ValueError` if the eligible pool is empty."""
    pool = _select_cells(adata, annot_col, annot_value, None)
    if len(pool) == 0:
        raise ValueError("resample_cells: eligible pool is empty")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pool), size=len(pool))
    return pool.to_numpy()[idx]


def subsample_gene_betas(adata, gene, factors=None, *, n_subsamples=100,
                         annot_col=None, annot_value=None, seed0=0,
                         metabolites='all', method='OLS', penalty=1.0, l1_ratio=0.5,
                         standardize=False, groups=None, onehot_col=None,
                         just_skip_samples_without_all_labels=False):
    """Beta of each selected factor for `gene` across `n_subsamples` cell resamples
    (seeds seed0..seed0+n_subsamples-1), each a same-size-with-replacement resample
    (see `resample_cells`) of the (optionally annotation-filtered) eligible cells.
    Returns a DataFrame of shape (n_subsamples, n_selected), columns = the selected
    column names, values = that column's fitted beta. NOT metabolite-specific:
    `factors` can be ANY modulator (bare TF gene, `lig$rec`, `lig#tf`, `metab@<name>`),
    plus (see below) a reduced `cluster_{id}` or a `{onehot_col}[{label}]` column.

    `groups` (default None): reduces the gene's factor matrix via `_reduce_groups`
    (see `fit_gene_betas`) -- same resulting `cluster_{id}` column(s), part of the
    column universe `factors`/`factors=None` selects from. To match `fit_gene_betas`'s
    `standardize=True` semantics (each constituent z-scored over the SAME population
    it's fit on), the reduction is redone PER RESAMPLE, on that resample's own drawn
    cells -- NOT once over the whole pool -- so a `cluster_*` beta from a resample
    whose drawn cells happen to equal some `fit_gene_betas(cells=...)` call matches
    that call's point estimate. The column set/order itself (`_reduce_groups` depends
    only on the gene's factor names + `groups`, not on which cells are selected) is
    fixed once from a template built over the full annotation-filtered pool.

    `onehot_col` (default None): fits an UNPENALIZED one-hot covariate via FWL (see
    `_fit`), exactly as in `fit_gene_betas`, in every resample's fit. The label set and
    reference are fixed ONCE from the full annotation-filtered pool (not per resample)
    so dummy columns stay consistent across draws. The fitted per-label coefficients
    (reference zeroed) are addressable as `f'{onehot_col}[{label}]'` in `factors`, and
    are included by default when `factors=None`. If a resample's drawn cells are
    missing any label: `just_skip_samples_without_all_labels=False` (default) raises
    `ValueError` naming the missing label(s); `True` records a NaN row for that
    resample and continues.

    `factors=None` -> the full column universe (the gene's -- possibly
    group-reduced -- factor columns, plus one `{onehot_col}[{label}]` per label when
    `onehot_col` is given); otherwise a subset (a list of exact column names; names not
    present are warned about and dropped). A resample drawing too few cells (fewer than
    the factor count, plus the onehot dummy columns and their shared intercept, plus 1)
    yields a NaN row -- only possible when the eligible pool itself is that small,
    since each resample is the same size as the pool. Builds the unreduced factor
    matrix (`X_raw`) and `y` ONCE and only row-slices per draw (efficiency); the
    reduction (when `groups` is given) and the small onehot `D` are rebuilt per draw,
    over that draw's own cells only (cheap -- z-score+sum / 0-1 columns over the drawn
    rows). The resample itself is drawn inline from a pool computed once (not via a
    per-iteration call to `resample_cells`, which would recompute the pool every time);
    behavior (same seed -> same draw) is identical to `resample_cells`.
    """
    X_raw = get_gene_factors(adata, gene, metabs=metabolites)
    pool = _select_cells(adata, annot_col, annot_value, None)
    if len(pool) == 0:
        raise ValueError("subsample_gene_betas: eligible pool is empty")
    pool_arr = pool.to_numpy()

    def _reduced(X_sub):
        return _reduce_groups(X_sub, groups, standardize) if groups is not None else X_sub

    # Column set/order is population-independent (depends only on X_raw's columns +
    # groups), so a template built over the full pool gives the universe/order that
    # every per-resample reduction will also produce.
    X_template = _reduced(X_raw.loc[pool])
    x_col_index = {name: i for i, name in enumerate(X_template.columns)}

    labels, dummy_labels, onehot_name_to_label, obs_series = [], [], {}, None
    if onehot_col is not None:
        obs_series = adata.obs[onehot_col].astype(str)
        labels = sorted(obs_series.loc[pool].unique())
        dummy_labels = labels[1:]
        onehot_name_to_label = {f'{onehot_col}[{lbl}]': lbl for lbl in labels}
    n_dummy_cols = len(dummy_labels)

    universe = list(X_template.columns) + list(onehot_name_to_label.keys())
    if factors is None:
        selected = list(universe)
    else:
        selected = [f for f in factors if f in universe]
        missing = [f for f in factors if f not in universe]
        if missing:
            warnings.warn(f"subsample_gene_betas: factors {missing} not in {gene!r}'s "
                          f"columns; dropped.")

    y = adata[:, gene].to_df(LAYER)[gene]
    min_cells = X_template.shape[1] + n_dummy_cols + 1

    out = np.full((n_subsamples, len(selected)), np.nan)
    for i, seed in enumerate(range(seed0, seed0 + n_subsamples)):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(pool_arr), size=len(pool_arr))
        cells = pool_arr[idx]
        if len(cells) < min_cells:
            continue

        D = None
        if onehot_col is not None:
            vals = obs_series.loc[cells].to_numpy()
            present = set(vals)
            missing_labels = [lbl for lbl in labels if lbl not in present]
            if missing_labels:
                if just_skip_samples_without_all_labels:
                    continue
                raise ValueError(
                    f"subsample_gene_betas: resample (seed={seed}) is missing "
                    f"label(s) {missing_labels} of {onehot_col!r}; set "
                    f"just_skip_samples_without_all_labels=True to skip resamples "
                    f"that don't contain every label of {onehot_col}.")
            D = (np.column_stack([(vals == lbl).astype(float) for lbl in dummy_labels])
                 if dummy_labels else np.zeros((len(cells), 0)))

        Xsub = _reduced(X_raw.loc[cells])
        beta, bD, _ = _fit(Xsub.to_numpy(), y.loc[cells].to_numpy(), method=method,
                           penalty=penalty, l1_ratio=l1_ratio, standardize=standardize, D=D)

        row = np.empty(len(selected))
        for j, f in enumerate(selected):
            if f in onehot_name_to_label:
                lbl = onehot_name_to_label[f]
                row[j] = 0.0 if lbl == labels[0] else bD[dummy_labels.index(lbl)]
            else:
                row[j] = beta[x_col_index[f]]
        out[i] = row
    return pd.DataFrame(out, columns=selected)


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
