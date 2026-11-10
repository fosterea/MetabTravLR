#!/usr/bin/env python
"""CLI entry point: compute + cache every signature-analysis result for one dataset config
(`signature_configs.CONFIGS`), so heavy compute runs on SLURM instead of in a notebook."""
import argparse
import sys
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
    compute_dataset(CONFIGS[args.config], overwrite=args.overwrite, targets=targets)
