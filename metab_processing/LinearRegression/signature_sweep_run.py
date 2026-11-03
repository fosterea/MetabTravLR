#!/usr/bin/env python
"""CLI entry point: compute + cache the penalty sweep (`signature_batch.
compute_sweep_dataset`) for one dataset config (`signature_configs.CONFIGS`) and one fit
method, so the heavy sweep compute runs on SLURM instead of in a notebook."""
import argparse
import sys
from pathlib import Path

_root = next((p for p in Path(__file__).resolve().parents
              if (p / '.git').exists() or (p / 'setup.py').exists()),
             Path(__file__).resolve().parent)
for _p in (str(_root), str(_root / 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metab_processing.LinearRegression.signature_batch import compute_sweep_dataset
from metab_processing.LinearRegression.signature_configs import CONFIGS

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(CONFIGS))
    ap.add_argument('--method', required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    if args.method not in cfg.sweep_penalties:
        ap.error(f"--method must be one of {sorted(cfg.sweep_penalties)} for "
                  f"--config {args.config!r}, got {args.method!r}")

    compute_sweep_dataset(cfg, args.method, overwrite=args.overwrite)
