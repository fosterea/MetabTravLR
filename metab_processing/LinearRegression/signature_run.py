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
    args = ap.parse_args()

    compute_dataset(CONFIGS[args.config], overwrite=args.overwrite)
