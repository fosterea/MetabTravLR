#!/usr/bin/env python
"""TEST harness: run SpaceTravLR with EVERY harreman transporter pair, not just the
significant ones, to see whether the model's own regularization regresses out the
extra noisy/insignificant metabolite pairs on its own.

A normal run sources its metabolites from `metabolite_selection.yaml` (harreman's
FDR-significant pairs only). This script instead sources them from
`metabolite_no_selection.yaml` (the RAW `gp_per_metabolite` network -- every transporter
pair harreman knows about, significant or not; written by
`metab_processing/Harreman/harreman_summary.py::write_metabolite_selection`). Everything
else about the pipeline is identical: same `run_spacetravlr.py` stages, same target genes,
same SLURM CLI. Comparing this run's betas (`all_metab_*` outputs) against the normal
selected-only run's betas answers the question: do the insignificant pairs end up with
~zero learned coefficient, or does the model actually use them?

This is a thin wrapper, not a fork: it reuses `run_spacetravlr` entirely by monkeypatching
its two module-global lookups --

  - `run_spacetravlr.dataset_paths`   (normally `dataset_configs.dataset_paths`)
  - `run_spacetravlr.load_metabolites` (normally `metab_loader.load_metabolites`)

-- with the two functions below, then calls `run_spacetravlr.main()` unmodified. Both
names are referenced as bare globals inside `run_spacetravlr.run_dataset`, so reassigning
them on the module redirects the whole pipeline without touching run_spacetravlr.py. Every
output path is redirected in parallel (`all_metab_*`), so a normal run is never touched.

This run is fully self-contained: `setup` (`input_data/`) is metabolite-agnostic and
identical to the normal run's, but because of the path isolation above it gets recomputed
under `all_metab_spacetravlr_output/` rather than shared with it -- the intentional price
of never touching a normal run's output. A user who wants to skip that recompute can pass
`--stage fit artifacts` once `all_metab_spacetravlr_output/input_data/` already exists.

    python run_all_metab.py --dataset Human_Lung
"""
import sys
from pathlib import Path

# Make the repo root (and src/) importable regardless of CWD or machine -- same idiom as
# run_spacetravlr.py.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parent,
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml

from metab_processing.metab_travlr_config import PROJECT_DATA_DIR
from metab_processing.SpaceTravLR import dataset_configs, run_spacetravlr
from metab_processing.SpaceTravLR.metab_loader import build_metabolites


def all_metab_paths(dataset, data_dir=PROJECT_DATA_DIR):
    """Same as `dataset_configs.dataset_paths`, but every output redirected to a parallel
    `all_metab_*` location and sourced from `metabolite_no_selection.yaml` instead of
    `metabolite_selection.yaml`. `adata`, `dataset_dir`, and `log_dir` are untouched.
    """
    paths = dataset_configs.dataset_paths(dataset, data_dir)
    paths['selection_yaml'] = (
        paths['dataset_dir'] / 'easy_download' / 'harreman_outputs' / 'metabolite_no_selection.yaml')
    paths['outdir'] = paths['dataset_dir'] / 'all_metab_spacetravlr_output'
    paths['input_data'] = paths['outdir'] / 'input_data'
    paths['betadata'] = paths['outdir'] / 'betadata'
    paths['metab_outdir'] = paths['dataset_dir'] / 'easy_download' / 'all_metab_metabtravlr_outputs'
    paths['beta_adata'] = paths['dataset_dir'] / 'all_metab_spacetravlr_adata.h5ad'
    return paths


def load_no_selection_metabolites(path, var_names=None, both_orientations=True):
    """Same signature/return as `metab_loader.load_metabolites`, but parses
    `metabolite_no_selection.yaml`'s DICT format (the raw `gp_per_metabolite` network:
    `{"metabolites": {name: {"gene_pair": [[g1,g2],...], ...}, ...}}`) instead of the
    normal selection file's LIST format.
    """
    with open(path) as f:
        doc = yaml.safe_load(f) or {}

    raw = doc.get('metabolites', {}) or {}
    selection = {}
    for name, info in raw.items():
        pairs = info.get('gene_pair', []) if isinstance(info, dict) else []
        selection[name] = [tuple(pair) for pair in pairs]

    metabolites = build_metabolites(selection, var_names=var_names, both_orientations=both_orientations)
    return metabolites, selection


def main(argv=None):
    print('=== ALL-METAB (no harreman selection) TEST run -- every transporter pair, not just significant ones ===')
    run_spacetravlr.dataset_paths = all_metab_paths
    run_spacetravlr.load_metabolites = load_no_selection_metabolites
    run_spacetravlr.main(argv)


if __name__ == '__main__':
    main()
