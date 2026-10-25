"""Shared compute surface for the signature analysis: build the per-(signature, run-spec)
result table ONCE, cache it to parquet, and let the viewer notebooks read the cache instead
of recomputing. Heavy compute (this module's `compute_dataset`) runs via `signature_run.py`
on SLURM; the notebooks only call `runspecs`/`load_res`.

No `src/SpaceTravLR/` changes. Depends only on numpy/pandas/scanpy + `signature_helpers`
(`sh`) + `signature_configs.DatasetConfig`.

NOTE -- cache layout: cached results (and the raw-betas sidecar) are keyed by
`(method, penalty, l1_ratio, standardize)` -- see `params_slug`/`cache_path` -- so each
distinct settings combo gets its own subdirectory under `cfg.cache_dir`. The OLD flat
`{cache_dir}/*.parquet` caches (written before this keying was added) are orphaned by this
change and can be deleted.
"""
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import scanpy as sc

from metab_processing.LinearRegression import signature_helpers as sh


def _slug(x):
    return re.sub(r'[^0-9A-Za-z]+', '_', str(x)).strip('_')


def _num(x):
    """Deterministic, filename-safe number -> slug: `format(x, 'g')` (so `0.005` and `5e-3`
    collapse to the same slug), then `.`->`p`, `-`->`m`, `+` dropped. E.g. `0.005`/`5e-3` ->
    `'0p005'`; `1e-05` -> `'1em05'`."""
    return format(x, 'g').replace('.', 'p').replace('-', 'm').replace('+', '')


def params_slug(method, penalty, l1_ratio, standardize):
    """The ONE place the cache settings-slug is built: `{method}_p{penalty}` +
    `_l{l1_ratio}` (elastic only) + `_nostd` (when standardize is False)."""
    slug = f"{method}_p{_num(penalty)}"
    if method == 'elastic':
        slug += f"_l{_num(l1_ratio)}"
    if not standardize:
        slug += "_nostd"
    return slug


def _log(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def runspecs(cfg):
    """Build the list of per-run specs for `cfg`: each a dict `{label, annot_col,
    annot_value, onehot_col, subset}` (`subset` is None or `(col, value)`). Labels are
    STABLE -- used both as cache-filename components and viewer-notebook titles."""
    specs = []
    if cfg.kind == 'cancer':
        if len(cfg.t_cell_types) > 1:
            specs.append({'label': 'all_pooled', 'annot_col': '_tcell_group',
                          'annot_value': 'GROUP', 'onehot_col': None, 'subset': None})
            specs.append({'label': f'all_pooled_cov_{cfg.cell_type_col}',
                          'annot_col': '_tcell_group', 'annot_value': 'GROUP',
                          'onehot_col': cfg.cell_type_col, 'subset': None})
        for ct in cfg.t_cell_types:
            specs.append({'label': f'{ct}_only', 'annot_col': cfg.cell_type_col,
                          'annot_value': ct, 'onehot_col': None, 'subset': None})
    elif cfg.kind == 'colon':
        for oh in [None, cfg.condition_col, cfg.patient_col, cfg.sample_col]:
            specs.append({'label': f"all__cov_{oh or 'none'}", 'annot_col': cfg.annot_col,
                          'annot_value': cfg.cell_type, 'onehot_col': oh, 'subset': None})
        for cond in ['UC', 'HC', 'ICI']:
            for oh in [None, cfg.patient_col, cfg.sample_col]:
                specs.append({'label': f"{cond}__cov_{oh or 'none'}",
                              'annot_col': cfg.annot_col, 'annot_value': cfg.cell_type,
                              'onehot_col': oh, 'subset': (cfg.condition_col, cond)})
    else:
        raise ValueError(f"runspecs: unknown cfg.kind {cfg.kind!r}")
    return specs


def all_targets(cfg):
    """The ordered list of target names cached/plotted for `cfg`: the colon raw single gene
    (if any), then the MSigDB target signatures, negative controls, and any MANUAL gene sets
    (`cfg.manual_sets`, resolved from a literal gene list rather than MSigDB). Shared by the
    compute loop and the viewer notebooks so they never drift."""
    base = [cfg.single_gene] if (cfg.kind == 'colon' and cfg.single_gene) else []
    return base + list(cfg.target_signatures) + list(cfg.neg_controls) + list(cfg.manual_sets)


def cache_path(cfg, signature, label, *, method=None, penalty=None, l1_ratio=None,
              standardize=None):
    """Cache file for (signature, label) under the settings subdir `params_slug(method,
    penalty, l1_ratio, standardize)` -- any override left `None` resolves to `cfg`'s own
    value. So `_membership.json` (written directly to `cfg.cache_dir`) stays
    penalty-independent, while every (method, penalty, l1_ratio, standardize) combo gets
    its own subdirectory of result/betas parquets."""
    method = cfg.method if method is None else method
    penalty = cfg.penalty if penalty is None else penalty
    l1_ratio = cfg.l1_ratio if l1_ratio is None else l1_ratio
    standardize = cfg.standardize if standardize is None else standardize
    return (Path(cfg.cache_dir) / params_slug(method, penalty, l1_ratio, standardize)
            / f'{_slug(signature)}__{_slug(label)}.parquet')


def betas_path(cfg, signature, label, **ov):
    """The raw-betas sidecar for (signature, label) (the n_sub x factor subsample-beta
    DataFrame), next to the result parquet from `cache_path`."""
    p = cache_path(cfg, signature, label, **ov)
    return p.with_name(p.stem + '.betas.parquet')


def load_res(cfg, signature, label, **ov):
    """Read a cached result, or None if not yet computed."""
    p = cache_path(cfg, signature, label, **ov)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_betas(cfg, signature, label, **ov):
    """Read the cached raw-betas sidecar, or None if not yet computed."""
    p = betas_path(cfg, signature, label, **ov)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _clear_outputs(cfg, signature, label):
    """Remove any stale res + betas parquets for a combo, so a combo that is being marked
    `.skip` (missing/failed) can never leave a prior success's data behind -- `_combo_done`
    would otherwise be satisfied by the stale res and serve it as 'cached'."""
    for f in (cache_path(cfg, signature, label), betas_path(cfg, signature, label)):
        if f.exists():
            f.unlink()


def _skip_marker(cfg, signature, label, **ov):
    """Sidecar marking a combo that was TRIED but can't produce a parquet (set missing from
    MSigDB, or the fit failed). Lets the short-circuit treat it as 'done' so a resubmitted
    job doesn't re-pay the adata load + factor diffusion just to re-skip it."""
    return cache_path(cfg, signature, label, **ov).with_suffix('.skip')


def _combo_done(cfg, signature, label):
    return cache_path(cfg, signature, label).exists() or _skip_marker(cfg, signature, label).exists()


def _atomic_write(path, text=None, df=None):
    """Write `text` (str) or `df` (parquet) via a tmp file + os.replace, so a job killed
    mid-write never leaves a truncated-but-existing file the short-circuit would trust.
    Creates the parent directory (e.g. the settings subdir from `cache_path`) lazily, so no
    other call site needs an explicit mkdir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    if df is not None:
        df.to_parquet(tmp)
    else:
        tmp.write_text(text if text is not None else '')
    os.replace(tmp, path)


def compute_run(adata, name, genes, spec, cfg, raw=False):
    """Run one (signature-or-gene, run-spec) fit + subsample on `adata` and return
    `(res, betas)`: the ranked/percentiled result DataFrame, and the raw `n_sub x factor`
    subsample-beta DataFrame (`sub` from `subsample_gene_betas(..., return_r=True)`) it was
    built from -- the same object `compute_dataset` caches as the `.betas.parquet` sidecar."""
    ad = adata
    if spec['subset'] is not None:
        col, val = spec['subset']
        ad = adata[adata.obs[col].astype(str) == str(val)].copy()

    if raw:
        ad2, target = ad, name
    else:
        ad2 = sh.add_gene_signature(ad, name, positive=genes, keep_when_overlap='factor',
                                    verbose=False)
        target = name

    b = sh.fit_gene_betas(ad2, genes=[target], annot_col=spec['annot_col'],
                          annot_value=spec['annot_value'], metabolites=[], custom='all',
                          method=cfg.method, penalty=cfg.penalty, l1_ratio=cfg.l1_ratio,
                          standardize=cfg.standardize, onehot_col=spec['onehot_col'])
    sub, sub_r = sh.subsample_gene_betas(
        ad2, target, factors=None, n_subsamples=cfg.n_sub, annot_col=spec['annot_col'],
        annot_value=spec['annot_value'], metabolites=[], custom='all', method=cfg.method,
        penalty=cfg.penalty, l1_ratio=cfg.l1_ratio, standardize=cfg.standardize,
        onehot_col=spec['onehot_col'], return_r=True,
        just_skip_samples_without_all_labels=True)
    res = sh.add_subsample_percentiles(sh.rank_coefficients(b), sub, sub_r,
                                       cfg.lower, cfg.upper).reset_index(drop=True)
    return res, sub


def compute_dataset(cfg, overwrite=False):
    """Compute + cache every (target, run-spec) result for `cfg`, skipping work already
    cached unless `overwrite=True`."""
    target_names = all_targets(cfg)
    raw_names = {cfg.single_gene} if (cfg.kind == 'colon' and cfg.single_gene) else set()
    specs = runspecs(cfg)

    # Guard against two distinct (signature, label) pairs slugging to the same cache file.
    combos = [(name, spec['label']) for name in target_names for spec in specs]
    fnames = [cache_path(cfg, name, label).name for name, label in combos]
    if len(set(fnames)) != len(fnames):
        dupes = sorted({f for f in fnames if fnames.count(f) > 1})
        raise ValueError(f'{cfg.name}: cache filename collision from _slug: {dupes}')

    # Short-circuit a fully-done dataset BEFORE loading adata / fetching MSigDB / diffusing
    # factors. A combo is "done" if its parquet OR a .skip marker exists -- so a dataset with
    # a permanently-missing set or a persistently-failing combo still short-circuits.
    membership_path = Path(cfg.cache_dir) / '_membership.json'
    if not overwrite and membership_path.exists() and all(
            _combo_done(cfg, name, label) for name, label in combos):
        _log('all cached, nothing to do')
        return

    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    _log(f'{cfg.name}: reading {cfg.adata_path}')
    adata = sc.read_h5ad(cfg.adata_path)
    var = set(adata.var_names)
    _log(f'{cfg.name}: {adata.n_obs} cells, {adata.n_vars} genes')
    library = sh.load_msigdb(cfg.dbver, cfg.categories)

    membership = {}
    for name in cfg.target_signatures + cfg.neg_controls:
        genes = library.get(name)
        if genes is None:
            _log(f'{cfg.name}: {name!r} not found in MSigDB; skipping at compute')
            membership[name] = {'present': [], 'missing': [], 'total': 0}
            continue
        present = [g for g in genes if g in var]
        missing = [g for g in genes if g not in var]
        removed, remaining = sh.signature_factor_split(adata, present)
        membership[name] = {'present': present, 'missing': missing, 'total': len(genes),
                            'factor_removed': removed, 'remaining': remaining}

    # Manual (non-MSigDB) target sets: resolve from the literal gene list.
    for name, genes in cfg.manual_sets.items():
        present = [g for g in genes if g in var]
        missing = [g for g in genes if g not in var]
        removed, remaining = sh.signature_factor_split(adata, present)
        membership[name] = {'present': present, 'missing': missing, 'total': len(genes),
                            'factor_removed': removed, 'remaining': remaining}

    for name in cfg.cancer_factors:
        genes = library.get(name)
        if genes is None:
            _log(f'{cfg.name}: {name!r} not found in MSigDB; skipping at compute')
            membership[name] = {'present': [], 'missing': [], 'total': 0}
            continue
        present = [g for g in genes if g in var]
        missing = [g for g in genes if g not in var]
        membership[name] = {'present': present, 'missing': missing, 'total': len(genes)}
    for vname, gl in cfg.vessel.items():
        present = [g for g in gl if g in var]
        missing = [g for g in gl if g not in var]
        membership[vname] = {'present': present, 'missing': missing, 'total': len(gl)}

    membership_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(membership_path, text=json.dumps(membership, indent=2))

    if cfg.kind == 'cancer':
        cancer_sets = {name: [g for g in library.get(name, []) if g in var]
                       for name in cfg.cancer_factors}
        cancer_sets.update({vname: [g for g in gl if g in var]
                            for vname, gl in cfg.vessel.items()})
        cancer_sets = {k: v for k, v in cancer_sets.items() if v}
        sh.add_cancer_factors(adata, cancer_sets, annot_col=cfg.cell_type_col,
                              add_proximity=True)
        if len(cfg.t_cell_types) > 1:
            sh.make_group_column(adata, cfg.cell_type_col, cfg.t_cell_types,
                                 group_name='_tcell_group', group_label='GROUP')

    sets = {name: [g for g in library.get(name, []) if g in var]
            for name in cfg.target_signatures + cfg.neg_controls}
    sets.update({name: [g for g in genes if g in var]
                 for name, genes in cfg.manual_sets.items()})

    n_done, n_failed, n_missing, n_cached = 0, 0, 0, 0
    for name in target_names:
        raw = name in raw_names
        # A manual set is a valid target even though it's absent from the MSigDB library.
        missing_set = (not raw) and (name not in library) and (name not in cfg.manual_sets)
        for spec in specs:
            p = cache_path(cfg, name, spec['label'])
            m = _skip_marker(cfg, name, spec['label'])
            if not overwrite and (p.exists() or m.exists()):
                _log(f'{cfg.name}: CACHED, skip {name} | {spec["label"]}')
                n_cached += 1
                continue
            if overwrite and m.exists():
                m.unlink()                       # redo: clear a stale skip marker
            if missing_set:
                _clear_outputs(cfg, name, spec['label'])     # don't leave stale success behind
                _atomic_write(m, text=f'missing from MSigDB: {name}')
                _log(f'{cfg.name}: SKIP (not in MSigDB) {name} | {spec["label"]}')
                n_missing += 1
                continue
            _log(f'{cfg.name}: computing {name} | {spec["label"]}')
            t0 = time.time()
            try:
                res, betas = compute_run(adata, name, sets.get(name, []), spec, cfg, raw=raw)
            except Exception as e:
                _clear_outputs(cfg, name, spec['label'])     # a now-failing combo must not serve stale data
                _atomic_write(m, text=f'{type(e).__name__}: {e}')   # mark failed -> short-circuitable
                _log(f'{cfg.name}: FAILED {name} | {spec["label"]}: {type(e).__name__}: {e}')
                n_failed += 1
                continue
            # Write betas FIRST then res, so `res.exists() => betas.exists()` always holds
            # (the short-circuit / _combo_done checks only the res parquet).
            _atomic_write(betas_path(cfg, name, spec['label']), df=betas)
            _atomic_write(p, df=res)
            if m.exists():
                m.unlink()                       # success supersedes any prior skip marker
            n_done += 1
            _log(f'{cfg.name}: wrote {p.name} ({len(res)} rows, {time.time() - t0:.1f}s)')

    _log(f'{cfg.name}: done -- {n_done} computed, {n_cached} cached, '
        f'{n_failed} failed, {n_missing} missing from MSigDB')
