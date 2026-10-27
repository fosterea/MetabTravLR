#!/usr/bin/env python
"""ONE-OFF: relabel the OLD flat signature cache into the new settings-keyed layout so it
is not orphaned (no recompute needed). RUN ONCE, then delete this file.

Old layout:  {cache_dir}/{sig}__{label}.parquet        (+ .skip markers)
New layout:  {cache_dir}/{params_slug}/{sig}__{label}.parquet

For each dataset in CONFIGS, this moves every parquet / .skip file sitting DIRECTLY in
`{cache_dir}` (i.e. the old flat files) into `{cache_dir}/{params_slug(cfg)}/`.
`_membership.json` stays at the root and is never moved; files already inside a subdir are
left alone, so re-running is safe/idempotent.

ASSUMPTION: the flat caches were computed with each config's CURRENT `method/penalty/
l1_ratio/standardize`. If you changed those settings since computing, the slug won't match
what was actually used -- fix the config (or move by hand) before running.

Usage (from the repo root or anywhere):
    python metab_processing/LinearRegression/migrate_signature_cache.py            # dry run (prints only)
    python metab_processing/LinearRegression/migrate_signature_cache.py --apply    # actually move
"""
import argparse
import sys
from pathlib import Path

_root = next((p for p in Path(__file__).resolve().parents
              if (p / '.git').exists() or (p / 'setup.py').exists()),
             Path(__file__).resolve().parents[2])
for _p in (str(_root), str(_root / 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metab_processing.LinearRegression.signature_configs import CONFIGS
from metab_processing.LinearRegression.signature_batch import params_slug


def migrate(apply=False):
    for key, cfg in CONFIGS.items():
        cdir = Path(cfg.cache_dir)
        if not cdir.exists():
            print(f'{key}: no cache dir ({cdir}); skip')
            continue
        slug = params_slug(cfg.method, cfg.penalty, cfg.l1_ratio, cfg.standardize)
        dest = cdir / slug
        # Only top-level files (not already in a subdir); never the membership json.
        movers = [f for f in cdir.iterdir()
                  if f.is_file() and f.name != '_membership.json'
                  and (f.suffix == '.parquet' or f.name.endswith('.skip'))]
        print(f'{key}: {len(movers)} flat file(s) -> {dest}/')
        for f in movers:
            target = dest / f.name
            if apply:
                dest.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    print(f'    SKIP (exists in dest): {f.name}')
                    continue
                f.rename(target)
            print(f'    {"moved" if apply else "would move"}: {f.name}')
    if not apply:
        print('\n(dry run -- re-run with --apply to actually move the files)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='actually move files (default: dry run)')
    migrate(apply=ap.parse_args().apply)
