"""Shared compute surface for the signature analysis: build the per-(signature, run-spec)
result table ONCE, cache it to parquet, and let the viewer notebooks read the cache instead
of recomputing. Heavy compute (this module's `compute_dataset`) runs via `signature_run.py`
on SLURM; the notebooks only call `runspecs`/`load_res`.

No `src/SpaceTravLR/` changes. Depends only on numpy/pandas/scanpy + `signature_helpers`
(`sh`) + `signature_configs.DatasetConfig`.
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


def cache_path(cfg, signature, label):
    return Path(cfg.cache_dir) / f'{_slug(signature)}__{_slug(label)}.parquet'


def load_res(cfg, signature, label):
    """Read a cached result, or None if not yet computed."""
    p = cache_path(cfg, signature, label)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _skip_marker(cfg, signature, label):
    """Sidecar marking a combo that was TRIED but can't produce a parquet (set missing from
    MSigDB, or the fit failed). Lets the short-circuit treat it as 'done' so a resubmitted
    job doesn't re-pay the adata load + factor diffusion just to re-skip it."""
    return cache_path(cfg, signature, label).with_suffix('.skip')


def _combo_done(cfg, signature, label):
    return cache_path(cfg, signature, label).exists() or _skip_marker(cfg, signature, label).exists()


def _atomic_write(path, text=None, df=None):
    """Write `text` (str) or `df` (parquet) via a tmp file + os.replace, so a job killed
    mid-write never leaves a truncated-but-existing file the short-circuit would trust."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    if df is not None:
        df.to_parquet(tmp)
    else:
        tmp.write_text(text if text is not None else '')
    os.replace(tmp, path)


def compute_run(adata, name, genes, spec, cfg, raw=False):
    """Run one (signature-or-gene, run-spec) fit + subsample on `adata` and return the
    ranked/percentiled result DataFrame."""
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
                          method=cfg.method, penalty=cfg.penalty, standardize=cfg.standardize,
                          onehot_col=spec['onehot_col'])
    sub, sub_r = sh.subsample_gene_betas(
        ad2, target, factors=None, n_subsamples=cfg.n_sub, annot_col=spec['annot_col'],
        annot_value=spec['annot_value'], metabolites=[], custom='all', method=cfg.method,
        penalty=cfg.penalty, standardize=cfg.standardize, onehot_col=spec['onehot_col'],
        return_r=True, just_skip_samples_without_all_labels=True)
    res = sh.add_subsample_percentiles(sh.rank_coefficients(b), sub, sub_r,
                                       cfg.lower, cfg.upper).reset_index(drop=True)
    return res


def compute_dataset(cfg, overwrite=False):
    """Compute + cache every (target, run-spec) result for `cfg`, skipping work already
    cached unless `overwrite=True`."""
    target_names = (([cfg.single_gene] if cfg.kind == 'colon' else [])
                    + list(cfg.target_signatures) + list(cfg.neg_controls))
    raw_names = {cfg.single_gene} if cfg.kind == 'colon' else set()
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

    n_done, n_failed, n_missing, n_cached = 0, 0, 0, 0
    for name in target_names:
        raw = name in raw_names
        missing_set = (not raw) and (name not in library)
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
                _atomic_write(m, text=f'missing from MSigDB: {name}')
                _log(f'{cfg.name}: SKIP (not in MSigDB) {name} | {spec["label"]}')
                n_missing += 1
                continue
            _log(f'{cfg.name}: computing {name} | {spec["label"]}')
            t0 = time.time()
            try:
                res = compute_run(adata, name, sets.get(name, []), spec, cfg, raw=raw)
            except Exception as e:
                _atomic_write(m, text=f'{type(e).__name__}: {e}')   # mark failed -> short-circuitable
                _log(f'{cfg.name}: FAILED {name} | {spec["label"]}: {type(e).__name__}: {e}')
                n_failed += 1
                continue
            _atomic_write(p, df=res)
            if m.exists():
                m.unlink()                       # success supersedes any prior skip marker
            n_done += 1
            _log(f'{cfg.name}: wrote {p.name} ({len(res)} rows, {time.time() - t0:.1f}s)')

    _log(f'{cfg.name}: done -- {n_done} computed, {n_cached} cached, '
        f'{n_failed} failed, {n_missing} missing from MSigDB')
