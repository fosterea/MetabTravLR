#!/usr/bin/env python
"""CLI entry point: compute + cache every signature-analysis result for one dataset config
(`signature_configs.CONFIGS`), so heavy compute runs on SLURM instead of in a notebook."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

_root = next((p for p in Path(__file__).resolve().parents
              if (p / '.git').exists() or (p / 'setup.py').exists()),
             Path(__file__).resolve().parent)
for _p in (str(_root), str(_root / 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metab_processing.LinearRegression.signature_batch import compute_dataset
from metab_processing.LinearRegression.signature_configs import CONFIGS

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(CONFIGS))
    ap.add_argument('--factor-group', default='nbhd', choices=['nbhd', 'slc', 'both'],
                    help="factor block: cancer neighborhood ('nbhd'), cell-autonomous SLC "
                         "genes ('slc'), or BOTH ('both')")
    ap.add_argument('--y-transform', default='normalized', choices=['normalized', 'cpm'],
                    help="target y: log1p(raw) 'normalized' (default) or log1p(CPM) 'cpm'. "
                         "Caches separately ({...}/y_cpm), never overwriting the default run.")
    ap.add_argument('--covariates', default=None,
                    help="comma-separated extra per-cell covariates to add as predictors: "
                         "subset of n_total_genes,n_total_transcripts (default none). "
                         "Caches separately ({...}/cov_...).")
    ap.add_argument('--exclude-factor-groups', default=None,
                    help="comma-separated factor groups to drop from the x design: subset "
                         "of tf,lr,ltf (default none). E.g. 'tf,ltf' keeps only LR factors "
                         "+ any custom block. Caches separately ({...}/excl_...).")
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--targets', default=None,
                    help='comma-separated subset of all_targets(cfg) to compute in THIS job '
                         '(for fanning one dataset across concurrent jobs); default = all')
    args = ap.parse_args()

    targets = None
    if args.targets is not None:
        targets = [t for t in args.targets.split(',') if t]
        if not targets:                      # '' or ',,' -> empty: a mistake, not "all"
            ap.error('--targets was empty after parsing; omit it to compute all targets')
    covariates = None
    if args.covariates is not None:
        covariates = tuple(c for c in args.covariates.split(',') if c)
        if not covariates:                   # '' or ',,' -> empty: a mistake, not "none"
            ap.error('--covariates was empty after parsing; omit it for no covariates')
    exclude_factor_groups = None
    if args.exclude_factor_groups is not None:
        exclude_factor_groups = tuple(g for g in args.exclude_factor_groups.split(',') if g)
        if not exclude_factor_groups:        # '' or ',,' -> empty: a mistake, not "none"
            ap.error('--exclude-factor-groups was empty after parsing; omit it for none')
        bad = [g for g in exclude_factor_groups if g not in ('tf', 'lr', 'ltf')]
        if bad:
            ap.error(f'--exclude-factor-groups: unknown group(s) {bad}; valid = tf,lr,ltf')

    cfg = CONFIGS[args.config]
    overrides = {}
    if args.factor_group != cfg.factor_group:
        overrides['factor_group'] = args.factor_group
    if args.y_transform != cfg.y_transform:
        overrides['y_transform'] = args.y_transform
    if covariates is not None:
        overrides['covariates'] = covariates
    if exclude_factor_groups is not None:
        overrides['exclude_factor_groups'] = exclude_factor_groups
    if overrides:
        cfg = replace(cfg, **overrides)
    compute_dataset(cfg, overwrite=args.overwrite, targets=targets)
