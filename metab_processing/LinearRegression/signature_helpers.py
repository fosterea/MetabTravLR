"""Shared helpers for the signature-analysis notebooks (melanoma / cervical / breast /
colon). Keeps the notebooks thin: MSigDB loading + membership reporting, the
subsample-percentile table, the interval/ratio diagnostic plots, adding cancer gene-set
signatures as spatial-neighborhood FACTORS, and a multi-run driver that predicts a
target signature across cell-type groupings (pooled / pooled+covariate / per-type).

Pure-analysis layer: numpy / pandas / matplotlib / seaborn / adjustText / gseapy, plus our
own `build_x` (no torch). `display` falls back to `print` outside a notebook.
"""
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

from metab_processing.LinearRegression.build_x import (
    add_gene_signature, add_neighborhood_scores, _factor_genes,
)
from metab_processing.LinearRegression.least_squares import (
    fit_gene_betas, subsample_gene_betas, rank_coefficients,
)

try:
    from IPython.display import display
except Exception:  # not in a notebook
    display = print


# ---------------------------------------------------------------------------------
# MSigDB loading + membership reporting
# ---------------------------------------------------------------------------------
# The categories the project's notebooks use, extended with reactome/wikipathways so the
# metabolic (TCA / pentose / folate / purine) and breast-cancer sets resolve.
DEFAULT_CATEGORIES = ('h.all', 'c2.cp.kegg_legacy', 'c2.cp.reactome',
                      'c2.cp.wikipathways', 'c5.go.bp')


def load_msigdb(dbver='2024.1.Hs', categories=DEFAULT_CATEGORIES):
    """Return {set_name: [genes]} unioned over `categories` of MSigDB `dbver`."""
    import gseapy as gp
    msig = gp.Msigdb()
    library = {}
    for cat in categories:
        try:
            library.update(msig.get_gmt(category=cat, dbver=dbver) or {})
        except Exception as e:
            print(f'{cat}: FAILED ({e})')
    return library


def resolve_sets(signatures, library, var, label='signatures'):
    """For each name in `signatures`: print full membership, what's MISSING from the data,
    and what's PRESENT. Returns {name: [present genes]} (what a pseudo-gene is built from).
    `var` is `set(adata.var_names)`."""
    sets = {}
    print(f'===== {label}: membership vs data ({len(signatures)} sets) =====')
    for name in signatures:
        genes = library.get(name)
        if genes is None:
            print(f'\n[MISSING SET -- not found in MSigDB]: {name}')
            continue
        present = [g for g in genes if g in var]
        absent = [g for g in genes if g not in var]
        sets[name] = present
        print(f'\n{name}  ({len(present)}/{len(genes)} present)')
        print(f'  in signature ({len(genes)}): {list(genes)}')
        print(f'  missing from data ({len(absent)}): {absent}')
        print(f'  present in data ({len(present)}): {present}')
    return sets


def signature_factor_split(adata, present_genes):
    """Mirror `add_gene_signature(keep_when_overlap='factor')`: a present signature gene is
    REMOVED if it is referenced by one of the signature's own factor columns (its union of
    constituent `x_factor_map` lists), else it REMAINS in the scored target. Returns
    (removed, remaining)."""
    fmap = adata.uns.get('x_factor_map', {})
    factors = set()
    for g in present_genes:
        factors.update(fmap.get(g, []))
    referenced = set()
    for f in factors:
        referenced.update(_factor_genes(f))
    removed = [g for g in present_genes if g in referenced]
    remaining = [g for g in present_genes if g not in referenced]
    return removed, remaining


def report_factor_removal(adata, sets):
    """Print, per target signature, which PRESENT genes are dropped for being referenced by
    a factor (as `keep_when_overlap='factor'` does) and which remain in the scored target."""
    print('===== target signatures: genes removed for being a factor =====')
    for name, present in sets.items():
        removed, remaining = signature_factor_split(adata, present)
        print(f'\n{name}')
        print(f'  removed (in factor set, {len(removed)}): {removed}')
        print(f'  remaining in target ({len(remaining)}): {remaining}')


# ---------------------------------------------------------------------------------
# Subsample-percentile table (matches the notebooks' inline helper)
# ---------------------------------------------------------------------------------
def add_subsample_percentiles(betas_df, sub, sub_r=None, lower=15, upper=85):
    """Add subsample-beta summary columns -- and, when `sub_r` is given, subsample-r columns
    -- to betas_df, matched by factor. `sub` / `sub_r` are the two n_subsamples x factor
    DataFrames from `subsample_gene_betas(..., return_r=True)`."""
    out = betas_df.copy()

    def stat(df, fn):
        return [fn(df[f]) if f in df.columns else np.nan for f in out['factor']]

    out['min_subsample_beta'] = stat(sub, np.nanmin)
    out['max_subsample_beta'] = stat(sub, np.nanmax)
    out['avg_subsample_beta'] = stat(sub, np.nanmean)
    out['var_subsample_beta'] = stat(sub, np.nanvar)
    out['meansq_subsample_beta'] = stat(sub, lambda s: np.nanmean(np.square(s)))  # mean(beta^2)
    out['msd_subsample_beta'] = [np.nanmean((sub[f] - b0) ** 2) if f in sub.columns else np.nan
                                 for f, b0 in zip(out['factor'], out['beta'])]
    out[f'lower_{lower}_subsample_beta'] = stat(sub, lambda s: np.nanpercentile(s, lower))
    out[f'upper_{upper}_subsample_beta'] = stat(sub, lambda s: np.nanpercentile(s, upper))
    if sub_r is not None:
        out['min_subsample_r'] = stat(sub_r, np.nanmin)
        out['max_subsample_r'] = stat(sub_r, np.nanmax)
        out['avg_subsample_r'] = stat(sub_r, np.nanmean)
    return out


def filter_nonzero_interval(res, lo_col='min_subsample_beta', hi_col='max_subsample_beta',
                            touch_zero=False):
    """Keep rows whose subsample beta interval [min, max] excludes zero (sign-stable effect).
    touch_zero=False: drop intervals that strictly cross 0 (touching allowed).
    touch_zero=True: also drop intervals that touch 0 (strictly one-sided only).
    NaN-bound rows are dropped."""
    lo, hi = res[lo_col], res[hi_col]
    keep = (lo > 0) | (hi < 0) if touch_zero else (lo >= 0) | (hi <= 0)
    return res[keep].copy()


# ---------------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------------
def plot_r_vs_beta(res, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(res['beta'], res['r'], s=12)
    ax.set_xlabel('beta'); ax.set_ylabel('r')
    return ax


def plot_interval_vs_beta(res, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    interval = res['max_subsample_beta'] - res['min_subsample_beta']
    ax.scatter(res['beta'], interval, s=12)
    x = np.linspace(res['beta'].min(), res['beta'].max(), 200)
    ax.plot(x, np.abs(x), color='red', linestyle='--', label='y = |x|')
    ax.set_xlabel('beta'); ax.set_ylabel('max - min subsample beta'); ax.legend()
    return ax


def plot_msd_vs_beta(res, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(res['beta'], res['msd_subsample_beta'], s=12)
    ax.set_xlabel('beta'); ax.set_ylabel('mean squared beta deviation')
    return ax


def _highlight_status(res, highlight_set):
    def is_highlighted(factor):
        genes = re.split(r'[#\$]', str(factor))
        return 'In Gene Set' if any(g in highlight_set for g in genes) else 'Other'
    return res['factor'].apply(is_highlighted)


def _ratio_scatter(res, ax, figsize, ylabel):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style='ticks')
    sns.scatterplot(data=res, x='x', y='y', hue='Gene Status',
                    palette={'In Gene Set': '#D55E00', 'Other': '#0072B2'},
                    s=25, alpha=0.8, ax=ax)
    texts = [ax.text(xi, yi, label, fontsize=8)
             for xi, yi, label in zip(res['x'], res['y'], res['factor'])]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                expand_points=(1.2, 1.2), expand_text=(1.2, 1.2))
    ax.set_xlabel('beta'); ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def filter_and_plot_ratio(res, ax=None, figsize=(10, 6), highlight_genes=None):
    """-log(interval / |avg_subsample_beta|) vs beta, over factors whose interval excludes 0
    (touch_zero). Signature genes highlighted."""
    highlight_set = set(highlight_genes or [])
    res = filter_nonzero_interval(res, touch_zero=True).copy()
    res['x'] = res['beta']
    interval = res['max_subsample_beta'] - res['min_subsample_beta']
    res['y'] = -np.log(interval / np.abs(res['avg_subsample_beta']))
    res['Gene Status'] = _highlight_status(res, highlight_set)
    return _ratio_scatter(res, ax, figsize, 'log(|avg_subsample_beta| / interval)')


def filter_and_plot_variance_ratio(res, ax=None, figsize=(10, 6), highlight_genes=None):
    """-log(var_subsample_beta / |avg_subsample_beta|) vs beta, over factors whose interval
    excludes 0 (touch_zero). Signature genes highlighted."""
    highlight_set = set(highlight_genes or [])
    res = filter_nonzero_interval(res, touch_zero=True).copy()
    res['x'] = res['beta']
    res['y'] = -np.log(res['var_subsample_beta'] / np.abs(res['avg_subsample_beta']))
    res['Gene Status'] = _highlight_status(res, highlight_set)
    return _ratio_scatter(res, ax, figsize, 'log(|avg_subsample_beta| / subsample_variance)')


def filter_and_plot_meansq_variance_ratio(res, ax=None, figsize=(10, 6), highlight_genes=None):
    """mean(beta^2) / subsample variance vs beta, over factors whose interval excludes 0
    (touch_zero). High = the beta's second moment is dominated by its mean, not its spread
    (a stable, consistently-signed effect). Signature genes highlighted."""
    highlight_set = set(highlight_genes or [])
    res = filter_nonzero_interval(res, touch_zero=True).copy()
    res['x'] = res['beta']
    res['y'] = res['meansq_subsample_beta'] / res['var_subsample_beta']
    res['Gene Status'] = _highlight_status(res, highlight_set)
    return _ratio_scatter(res, ax, figsize, 'mean(beta^2) / subsample variance')


# ---------------------------------------------------------------------------------
# Cancer gene-set signatures as spatial-neighborhood FACTORS
# ---------------------------------------------------------------------------------
def add_cancer_factors(adata, sets, annot_col, add_proximity=True,
                       radius=300, scale_factor=1, eps=1e-3, verbose=True):
    """Add each cancer-associated gene-set signature in `sets` ({name: [present genes]}) to
    the adata's custom-factor block as a spatial-neighborhood score, SPLIT by `annot_col`
    (one `{celltype}-{SET}` factor per cell type), via `build_x.add_neighborhood_scores`.
    `add_proximity=True` also adds the kernel-weighted `{celltype}-proximity` density
    controls (so a signal that is just "this cell type is nearby" is separable). These
    become predictors for any target fit with `custom='all'`. Mutates/returns adata."""
    gene_sets = {name: genes for name, genes in sets.items() if genes}
    if not gene_sets:
        print('add_cancer_factors: no non-empty sets to add.'); return adata
    add_neighborhood_scores(adata, gene_sets, annot_col=annot_col,
                            add_proximity=add_proximity, radius=radius,
                            scale_factor=scale_factor, eps=eps)
    if verbose:
        n = len(adata.uns.get('x_custom_cols', []))
        print(f'add_cancer_factors: added/updated custom factors; '
              f"x_custom now has {n} columns from {len(gene_sets)} signature(s).")
    return adata


# ---------------------------------------------------------------------------------
# Multi-run signature driver (pooled / pooled+covariate / per cell-type)
# ---------------------------------------------------------------------------------
def make_group_column(adata, cell_type_col, group_values, group_name='_group',
                      group_label='GROUP'):
    """Write `adata.obs[group_name]` = `group_label` for cells whose `cell_type_col` is in
    `group_values`, else 'other'. Lets a fit select "all of these cell types pooled" via
    a single annot_value (fit_gene_betas matches one value, not a list)."""
    s = adata.obs[cell_type_col].astype(str)
    adata.obs[group_name] = np.where(s.isin([str(v) for v in group_values]),
                                     group_label, 'other')
    return group_name


def run_one(adata, name, sets, *, annot_col, annot_value, onehot_col=None, samples_name='',
            method='l1', penalty=0.005, standardize=True, n_sub=50, lower=15, upper=85,
            top=40, custom='all', metabolites=(), keep_when_overlap='factor',
            plots=('ratio', 'variance_ratio', 'meansq_variance_ratio')):
    """One signature fit: build the `name` pseudo-gene, fit on `annot_col==annot_value`
    cells with an optional unpenalized batch covariate `onehot_col`, resample, and plot the
    ratio diagnostics. Used by the colon notebook (fixed cell type, varying sample subset +
    batch covariate). Returns the res DataFrame (or None)."""
    if name not in sets:
        print(f'skip {name}: set not available'); return None
    genes = sets[name]
    ad2 = add_gene_signature(adata, name, positive=genes,
                             keep_when_overlap=keep_when_overlap, verbose=True)
    label = f'{samples_name} | {annot_value} | covariate={onehot_col}'
    print(f'\n===== {name} | {label} =====')
    try:
        b = fit_gene_betas(ad2, genes=[name], annot_col=annot_col, annot_value=annot_value,
                           metabolites=list(metabolites), custom=custom, method=method,
                           penalty=penalty, standardize=standardize, onehot_col=onehot_col)
        sub, sub_r = subsample_gene_betas(
            ad2, name, factors=None, n_subsamples=n_sub, annot_col=annot_col,
            annot_value=annot_value, metabolites=list(metabolites), custom=custom,
            method=method, penalty=penalty, standardize=standardize, onehot_col=onehot_col,
            return_r=True, just_skip_samples_without_all_labels=True)
    except Exception as e:
        print(f'  run failed ({type(e).__name__}: {e})'); return None
    res = add_subsample_percentiles(rank_coefficients(b), sub, sub_r, lower, upper)
    display(res.head(top))
    plot_res(res, genes, title=f'{name} | {label}', plots=plots)
    return res


def plot_res(res, highlight_genes=None, *, title='', plots=('ratio', 'variance_ratio',
                                                            'meansq_variance_ratio')):
    """Render the selected ratio diagnostics for an (already computed / loaded) res frame.
    The viewer notebooks call this on cached DataFrames."""
    if res is None or res.empty:
        print(f'(no rows to plot) {title}'); return
    specs = {'ratio': (filter_and_plot_ratio, '|avg_beta|/INTERVAL'),
             'variance_ratio': (filter_and_plot_variance_ratio, '|avg_beta|/VARIANCE'),
             'meansq_variance_ratio': (filter_and_plot_meansq_variance_ratio, 'mean(beta^2)/VARIANCE')}
    for key in plots:
        fn, tag = specs[key]
        fn(res, highlight_genes=highlight_genes).set_title(f'{title} | {tag}')


def run_signature(adata, name, sets, cell_types, *, cell_type_col='cell_type',
                  method='l1', penalty=0.005, standardize=True, n_sub=50,
                  lower=15, upper=85, top=40, custom='all', metabolites=(),
                  keep_when_overlap='factor',
                  plots=('ratio', 'variance_ratio', 'meansq_variance_ratio'),
                  highlight_full_set=None):
    """Predict target signature `name` (an all-positive pseudo-gene) across cell-type
    groupings and plot the ratio diagnostics for each run. Three kinds of run:
      1. all `cell_types` pooled (no covariate),
      2. all `cell_types` pooled WITH `cell_type_col` as an unpenalized one-hot covariate,
      3. each cell type in `cell_types` individually.
    `custom='all'` includes the cancer neighborhood factors as predictors. Returns
    {run_label: res_dataframe}.
    """
    if name not in sets:
        print(f'skip {name}: set not available'); return {}
    genes = sets[name]
    highlight = highlight_full_set if highlight_full_set is not None else genes
    ad2 = add_gene_signature(adata, name, positive=genes,
                             keep_when_overlap=keep_when_overlap, verbose=True)

    # Pooled / pooled+covariate runs only make sense with >1 cell type; with a single type
    # they collapse to the per-type run (e.g. breast's lone 'T Lymphocytes'), so skip them.
    runs = []
    if len(cell_types) > 1:
        grp_col = make_group_column(ad2, cell_type_col, cell_types,
                                    group_name='_tcell_group', group_label='GROUP')
        runs += [('all cell types (pooled)', grp_col, 'GROUP', None),
                 (f'all cell types (pooled) + {cell_type_col} covariate', grp_col, 'GROUP', cell_type_col)]
    runs += [(f'{ct} only', cell_type_col, ct, None) for ct in cell_types]

    results = {}
    for label, acol, aval, onehot in runs:
        print(f'\n===== {name} | {label} =====')
        try:
            b = fit_gene_betas(ad2, genes=[name], annot_col=acol, annot_value=aval,
                               metabolites=list(metabolites), custom=custom,
                               method=method, penalty=penalty, standardize=standardize,
                               onehot_col=onehot)
            sub, sub_r = subsample_gene_betas(
                ad2, name, factors=None, n_subsamples=n_sub, annot_col=acol, annot_value=aval,
                metabolites=list(metabolites), custom=custom, method=method, penalty=penalty,
                standardize=standardize, onehot_col=onehot, return_r=True,
                just_skip_samples_without_all_labels=True)
        except Exception as e:
            print(f'  run failed ({type(e).__name__}: {e})'); continue
        res = add_subsample_percentiles(rank_coefficients(b), sub, sub_r, lower, upper)
        results[label] = res
        display(res.head(top))
        plot_res(res, highlight, title=f'{name} | {label}', plots=plots)
    return results
