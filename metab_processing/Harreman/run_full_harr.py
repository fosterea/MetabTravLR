import sys
from pathlib import Path

# Make the repo root importable regardless of CWD or machine: walk up from this
# file until we hit a repo marker, then put that dir on sys.path.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parent,
)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


import os
import gc
import torch
from pathlib import Path

from harreman_funcs import HarremanRunner
import harreman_summary


from metab_processing.metab_travlr_config import PROJECT_DATA_DIR, DATA_DIR as METAB_DATA_DIR

XENIUM_DATA_DIR = PROJECT_DATA_DIR
TIERS = ['Tier1', 'Tier2', 'Tier3']


def run_dataset(data_dir, dataset_name, tiers=TIERS):
    harRunner = HarremanRunner(f'{data_dir}/{dataset_name}')
    harRunner.load_adata()
    harRunner.save_harreman_network()

    present = [tier for tier in tiers if tier in harRunner.adata.obs.columns]
    if not present:
        raise ValueError(f'none of {tiers} in adata.obs')

    for tier in present:
        harRunner.run_harreman(tier)

    out_path = harRunner.easy_download_path
    master, genepairs = harreman_summary.summarize_harreman_folder(out_path, sample_id=dataset_name)
    summary_dir = Path(out_path) / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(summary_dir / 'metabolite_summary.csv', index=False)
    genepairs.to_csv(summary_dir / 'gene_pair_summary.csv', index=False)
    harreman_summary.select_tcell_metabolites(out_path)

    # written last so a dataset is only skipped once it finished
    marker_path(data_dir, dataset_name).write_text(dataset_name)

def marker_path(data_dir, dataset_name):
    return Path(f'{data_dir}/{dataset_name}/easy_download/.{dataset_name}')


def run_all(data_dir=XENIUM_DATA_DIR, tiers=TIERS):
    d_sets = os.listdir(data_dir)
    # d_sets = ['Primary_Dermal_Melanoma']
    print(d_sets)
    for dataset_name in sorted(d_sets):
        if not os.path.isdir(f'{data_dir}/{dataset_name}'):
            continue
        if marker_path(data_dir, dataset_name).is_file():
            print(f'skipping {dataset_name}, already done')
            continue

        print(f'running {dataset_name}')
        try:
            run_dataset(data_dir, dataset_name, tiers=tiers)
            print(f'finished {dataset_name}')
        except Exception as e:
            print(f'failed {dataset_name}: {type(e).__name__}: {e}')

        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description='Run harreman over every dataset folder under a data dir.')
    ap.add_argument('--data-dir', default=XENIUM_DATA_DIR,
                    help='top-level dir whose subfolders are datasets (default: PROJECT_DATA_DIR)')
    ap.add_argument('--cell-type-cols', nargs='+', default=TIERS,
                    help='adata.obs column(s) to run cell-type-aware harreman over; '
                         'one output subfolder per column (default: Tier1 Tier2 Tier3)')
    args = ap.parse_args(argv)
    run_all(data_dir=args.data_dir, tiers=args.cell_type_cols)


if __name__ == '__main__':
    main()