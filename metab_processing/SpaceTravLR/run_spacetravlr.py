#!/usr/bin/env python
"""Run SpaceTravLR end-to-end for one dataset. This is the body of the SLURM job.

Stages (``--stage``, default all three, in this order):

  setup      SpaceShip.setup_  -> spacetravlr_output/input_data/{_adata.h5ad,
             celloracle_links.pkl, tflinks.parquet}. Skipped if already complete.
  fit        SpaceShip.fit(metabolites=...) -> spacetravlr_output/betadata/<gene>_betadata.parquet.
             Resumable: a gene whose parquet already exists is skipped by the queue,
             so re-running after a timeout picks up where it left off.
  artifacts  beta_analysis -> easy_download/metabtravlr_outputs/<tier>/{metabolites.csv,
             histograms.csv,histograms.png} and spacetravlr_adata.h5ad (per-cell betas
             attached to the full adata). CPU-only, cheap to re-run on its own.

Submit it with ``submit_spacetravlr.py`` (or the notebook) rather than calling it directly;
that is what creates the log directory and the sbatch job.

    python run_spacetravlr.py --dataset Primary_Dermal_Melanoma
    python run_spacetravlr.py --dataset Human_Lung --stage artifacts
    python run_spacetravlr.py --dataset Human_Lung --overwrite
"""
import sys
from pathlib import Path

# Make the repo root (and src/) importable regardless of CWD or machine: walk up from
# this file until we hit a repo marker, then put that dir on sys.path.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parent,
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import time

import h5py
import scanpy as sc

from SpaceTravLR.spaceship import SpaceShip
from metab_processing.metab_travlr_config import PROJECT_DATA_DIR
from metab_processing.SpaceTravLR import beta_analysis
from metab_processing.SpaceTravLR.dataset_configs import DATASETS, dataset_paths, get_config
from metab_processing.SpaceTravLR.metab_loader import load_metabolites

STAGES = ('setup', 'fit', 'artifacts')

# Everything `fit` needs from a finished setup.
_SETUP_ARTIFACTS = (
    'input_data/_adata.h5ad',
    'input_data/celloracle_links.pkl',
    'input_data/tflinks.parquet',
)

# Smallest cluster MAGIC will accept. `impute_clusterwise` (oracles.py:84) runs MAGIC once
# per cluster with the library defaults, and graphtools then asks sklearn for 16 neighbours
# -- observed as `ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 16,
# n_samples_fit = 10` on Human_Breast at leiden_scVI_res_0.5. There is no size guard in the
# package, and the crash lands only when the loop reaches that cluster, which on a large
# dataset is an hour of imputation thrown away. Check before setup starts instead.
MAGIC_MIN_CLUSTER_CELLS = 16

# Cells in sub-MAGIC clusters are dropped at load. A few outliers are fine to discard, but
# past this fraction of all cells the resolution itself is too fine -- stop and ask for a
# coarser one rather than silently throwing away a meaningful slice of the dataset.
MAX_TINY_CLUSTER_CELL_FRACTION = 0.02


def _log(msg):
    """Timestamped, flushed -- SLURM logs are otherwise block-buffered and useless live."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _isolate_cache_dir():
    """Move this process's cache dir onto node-local disk, before celloracle is imported.

    `import celloracle_tmp` (spaceship.py:264) pulls in genomepy, which AT IMPORT TIME opens a
    diskcache SQLite database under `appdirs.user_cache_dir()` -- i.e. `~/.cache/genomepy/<ver>`,
    which on Savio is NFS home. SQLite's locking is unreliable on NFS, so opening it raises
    `sqlite3.OperationalError: locking protocol`; every job also shares that one file, so
    concurrent jobs make it worse. diskcache only retries 'database is locked', not this.

    `/tmp` on a compute node is local disk (unlike `$TMPDIR`, which on some clusters points at
    network scratch and would have the same problem). The cache only memoizes genomepy provider
    metadata, which nothing in our pipeline uses, so starting cold costs nothing.
    """
    default = f'/tmp/spacetravlr_cache_{os.environ.get("SLURM_JOB_ID", "local")}'
    os.environ.setdefault('XDG_CACHE_HOME', default)
    os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)
    _log(f'XDG_CACHE_HOME={os.environ["XDG_CACHE_HOME"]}')


def _h5ad_is_readable(path) -> bool:
    """Cheap truncation check -- `write_h5ad` is not atomic, so a job killed mid-write
    (wall-time, OOM) leaves a file that exists but cannot be opened. Existence alone would
    make the resubmitted job skip setup and then crash deep inside `fit`.
    """
    try:
        with h5py.File(path, 'r') as f:
            return 'var' in f and 'obs' in f
    except Exception:
        return False


def setup_is_complete(outdir) -> bool:
    """True if a previous setup left everything the fit stage needs.

    Deliberately not `SpaceShip.is_everything_ok()`: that asserts a CWD-relative
    `launch.py` exists, which is never true inside a SLURM job.
    """
    outdir = Path(outdir)
    if not all((outdir / f).is_file() for f in _SETUP_ARTIFACTS):
        return False
    return _h5ad_is_readable(outdir / 'input_data' / '_adata.h5ad')


def _job_is_active(job_id) -> bool:
    """Is this SLURM job still queued or running?

    Used to tell a live lock from one orphaned by a job that died without unwinding
    (OOM kill and scancel are SIGKILL, so no `finally` runs).

    Only "squeue ran and listed nothing" counts as gone -- squeue reports a finished job
    as an invalid id, so we cannot also require exit status 0. A hand-run owner or an
    unavailable squeue counts as active, so we never steal a lock we cannot ask about.
    """
    if not job_id.isdigit():
        return True          # hand-run, not a job we can ask about
    try:
        done = subprocess.run(['squeue', '-h', '-j', job_id],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return True          # no squeue (off-cluster) -> assume alive
    return bool(done.stdout.strip())


def _read_lock_owner(lock) -> str:
    try:
        return lock.read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return ''


@contextlib.contextmanager
def _setup_lock(outdir):
    """Refuse to run two setups on the same dataset at once.

    The fit stage is already safe for concurrent workers (`oracles.py`'s lock-based gene
    queue), but two setups would write `_adata.h5ad` and `celloracle_links.pkl` to the same
    paths simultaneously -- HDF5 has no support for that and the result is a corrupt file.

    The lock records its owning job id, and a lock whose job is no longer in the queue is
    taken over rather than honoured -- otherwise a single OOM kill would block every
    resubmit until someone deleted the file by hand.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lock = outdir / '.setup.lock'
    owner = os.environ.get('SLURM_JOB_ID', f'local-{os.getpid()}')
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    try:
        fd = os.open(lock, flags)
    except FileExistsError:
        previous = _read_lock_owner(lock)
        if _job_is_active(previous):
            raise RuntimeError(
                f'{lock} is held by job {previous or "unknown"}, which is still active -- '
                f'a setup is already running for this dataset. Check `squeue`.'
            ) from None
        _log(f'clearing stale {lock} (job {previous or "unknown"} is no longer in the queue)')
        lock.unlink(missing_ok=True)
        try:
            fd = os.open(lock, flags)
        except FileExistsError:
            raise RuntimeError(f'{lock} was re-created concurrently; retry shortly') from None

    try:
        os.write(fd, f'{owner}\nstarted {time.strftime("%Y-%m-%d %H:%M:%S")}\n'.encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _trained_genes(paths):
    """Target genes that already have a betadata parquet (what `fit` would skip)."""
    if not paths['betadata'].exists():
        return []
    return sorted(p.name[:-len('_betadata.parquet')]
                  for p in paths['betadata'].glob('*_betadata.parquet'))


def _load_adata(paths, cell_type_src):
    """The raw dataset adata, prepared the way setup_ expects it.

    `cell_type` is the annotation SpaceShip clusters on; `raw_count` is required by
    `run_celloracle_` (spaceship.py:269).
    """
    _log(f'reading {paths["adata"]}')
    adata = sc.read_h5ad(paths['adata'])
    if cell_type_src not in adata.obs.columns:
        raise KeyError(
            f'cell_type_src {cell_type_src!r} not in adata.obs; '
            f'set it in dataset_configs.py. available: {list(adata.obs.columns)}'
        )
    adata.obs['cell_type'] = adata.obs[cell_type_src]
    adata.layers['raw_count'] = adata.X
    _log(f'adata: {adata.n_obs} cells x {adata.n_vars} genes')
    return adata


def _cluster_sizes(obs, column):
    """Observed cluster sizes in `column`, smallest first.

    Categorical columns keep categories with no cells; `impute_clusterwise` iterates
    `obs[annot].unique()`, which does not, so drop the empties or we would report
    phantom zero-cell clusters.
    """
    counts = obs[column].value_counts()
    return counts[counts > 0].sort_values()


def _sibling_annotations(obs, cell_type_src):
    """Other resolutions of the same annotation family.

    'leiden_scVI_res_0.5' -> every other 'leiden_scVI_res_*' column, coarsest first.
    Empty if `cell_type_src` has no trailing number to strip.
    """
    prefix = re.sub(r'[\d.]+$', '', cell_type_src)
    if prefix == cell_type_src:
        return []

    def _res(name):
        try:
            return float(name[len(prefix):])
        except ValueError:
            return float('inf')

    siblings = [c for c in obs.columns if c != cell_type_src and c.startswith(prefix)]
    return sorted(siblings, key=_res)


def _drop_tiny_clusters(adata, cell_type_src):
    """Drop cells in clusters too small for MAGIC and return the trimmed adata.

    `impute_clusterwise` runs MAGIC once per cluster and graphtools then asks sklearn for
    MAGIC_MIN_CLUSTER_CELLS neighbours; a smaller cluster crashes setup partway through
    (oracles.py:113). Such a cluster would also fit its own per-cluster GroupLasso on a
    handful of cells -- overfit betas we would not trust anyway -- so we discard those
    cells here rather than impute and train them. This only trims the in-memory copy that
    becomes the processed `_adata.h5ad`; the source `adata.h5ad` is never modified.

    Dropping is only right for a few outliers. Above MAX_TINY_CLUSTER_CELL_FRACTION of all
    cells the annotation is simply too fine -- raise and name a coarser resolution instead.
    """
    sizes = _cluster_sizes(adata.obs, cell_type_src)
    too_small = sizes[sizes < MAGIC_MIN_CLUSTER_CELLS]
    if too_small.empty:
        _log(f'{cell_type_src}: {len(sizes)} clusters, smallest {int(sizes.iloc[0])} '
             f'cells (>= {MAGIC_MIN_CLUSTER_CELLS}) -- keeping every cell')
        return adata

    n_drop = int(too_small.sum())
    frac = n_drop / adata.n_obs
    offenders = ', '.join(f'{c!r}={int(n)}' for c, n in too_small.items())

    if frac > MAX_TINY_CLUSTER_CELL_FRACTION:
        lines = [
            f'{cell_type_src}: {len(too_small)} cluster(s) below the '
            f'{MAGIC_MIN_CLUSTER_CELLS}-cell MAGIC minimum hold {n_drop} cells '
            f'({frac:.1%} of {adata.n_obs}) -- too many to drop as outliers ({offenders}). '
            f'Pick a coarser resolution via cell_type_src in dataset_configs.py.',
        ]
        siblings = _sibling_annotations(adata.obs, cell_type_src)
        if siblings:
            lines.append('  annotation                     clusters  smallest  usable')
            for col in siblings:
                other = _cluster_sizes(adata.obs, col)
                smallest = int(other.iloc[0]) if len(other) else 0
                ok = 'yes' if smallest >= MAGIC_MIN_CLUSTER_CELLS else 'no'
                lines.append(f'  {col:<30} {len(other):>8} {smallest:>9}  {ok}')
        raise ValueError('\n'.join(lines))

    _log(f'{cell_type_src}: dropping {n_drop} cell(s) ({frac:.2%} of {adata.n_obs}) in '
         f'{len(too_small)} cluster(s) below the {MAGIC_MIN_CLUSTER_CELLS}-cell MAGIC '
         f'minimum: {offenders}')
    keep = ~adata.obs[cell_type_src].isin(list(too_small.index))
    adata = adata[keep.values].copy()
    for col in {cell_type_src, 'cell_type'} & set(adata.obs.columns):
        if hasattr(adata.obs[col], 'cat'):
            adata.obs[col] = adata.obs[col].cat.remove_unused_categories()
    kept = _cluster_sizes(adata.obs, cell_type_src)
    _log(f'{cell_type_src}: kept {adata.n_obs} cells in {len(kept)} clusters, '
         f'smallest now {int(kept.iloc[0])}')
    return adata


def _processed_var_names(paths):
    """var_names of the *processed* adata, without loading it into memory.

    Used to drop metab pairs whose transporter is not in the panel. Read from the
    processed copy so the fit stage stands alone (no need to touch the raw adata).
    """
    backed = sc.read_h5ad(paths['input_data'] / '_adata.h5ad', backed='r')
    try:
        return list(backed.var_names)
    finally:
        if backed.isbacked:
            backed.file.close()


def run_dataset(dataset, stages=STAGES, overwrite=False, clear_betadata=False,
                data_dir=PROJECT_DATA_DIR):
    """Run `stages` for one dataset. See the module docstring for what each stage writes."""
    cfg = get_config(dataset)
    paths = dataset_paths(dataset, data_dir)
    focus_genes = cfg['focus_genes']

    _log(f'=== {dataset} | stages={list(stages)} | overwrite={overwrite} ===')
    _log(f'outdir: {paths["outdir"]}')
    _isolate_cache_dir()
    if overwrite and 'setup' not in stages:
        raise ValueError('--overwrite redoes setup, so it needs the setup stage; '
                         'got --stage without it. Use --clear-betadata to drop trained genes.')
    for key in ('adata', 'selection_yaml'):
        if not paths[key].exists():
            raise FileNotFoundError(f'missing {key}: {paths[key]}')

    ship = SpaceShip(name=dataset.replace('/', '_'), outdir=str(paths['outdir']), genes=focus_genes)
    adata = None

    # ------------------------------------------------- directory lifecycle
    # Done up front, not inside a stage, so `--stage fit --clear-betadata` actually retrains.
    trained_before = _trained_genes(paths)
    if overwrite and paths['input_data'].exists():
        _log(f'overwrite: removing {paths["input_data"]}')
        shutil.rmtree(paths['input_data'])
    if clear_betadata and paths['betadata'].exists():
        _log(f'clear-betadata: removing {paths["betadata"]} ({len(trained_before)} trained genes)')
        shutil.rmtree(paths['betadata'])
    elif overwrite and trained_before:
        _log(f'NOTE: keeping {len(trained_before)} existing betadata parquets -- they were trained '
             f'on the PREVIOUS preprocessing. Pass --clear-betadata for a clean slate.')

    # ---------------------------------------------------------------- setup
    if 'setup' in stages:
        if setup_is_complete(paths['outdir']):
            _log('setup already complete, skipping (--overwrite to redo)')
        else:
            with _setup_lock(paths['outdir']):
                adata = _load_adata(paths, cfg['cell_type_src'])
                adata = _drop_tiny_clusters(adata, cfg['cell_type_src'])
                _log(f'setup_ (run_commot={cfg["run_commot"]}) ...')
                # overwrite=True only bypasses setup_'s "directory exists" guard; it deletes
                # nothing. We own the directory lifecycle above.
                ship.setup_(adata, overwrite=True, run_commot=cfg['run_commot'])
                if not setup_is_complete(paths['outdir']):
                    missing = [f for f in _SETUP_ARTIFACTS if not (paths['outdir'] / f).is_file()]
                    raise RuntimeError(
                        f'setup_ finished but the result is unusable; missing: {missing or "none"} '
                        f'(if nothing is missing, _adata.h5ad did not open cleanly)')
            _log('setup complete')

    if 'fit' in stages:
        # Training reads its own `_adata.h5ad` from disk, so don't hold a full Xenium adata
        # in memory for the hours it runs. The artifacts stage reloads it.
        adata = None

    # ------------------------------------------------------------------ fit
    if 'fit' in stages:
        if not setup_is_complete(paths['outdir']):
            raise RuntimeError(f'setup is not complete for {dataset}; run --stage setup first')

        metabolites, selection = load_metabolites(
            paths['selection_yaml'], var_names=_processed_var_names(paths))
        _log(f'{len(selection)} metabolites -> {len(metabolites)} model columns')
        _log(f'fitting {len(focus_genes)} target genes: {focus_genes}')
        ship.fit(metabolites=metabolites, **cfg['fit_kwargs'])
        done = _trained_genes(paths)
        _log(f'fit complete; {len(done)} genes with betadata: {done}')
        untrained = [g for g in focus_genes if g not in set(done)]
        if untrained:
            _log(f'NOTE: {len(untrained)} target genes produced no betadata (orphaned -- no '
                 f'regulators and no metabolite modulators, or all-zero betas): {untrained}')

    # ------------------------------------------------------------ artifacts
    if 'artifacts' in stages:
        # Guard on the CURRENT target genes, not "any parquet at all" -- stale parquets from
        # a previous focus_genes would otherwise sail through and write empty CSVs.
        trained = _trained_genes(paths)
        usable = [g for g in focus_genes if g in set(trained)]
        if not usable:
            raise RuntimeError(
                f'no betadata for any of the {len(focus_genes)} target genes {focus_genes} in '
                f'{paths["betadata"]}; found {trained or "nothing"}. Run --stage fit first, or '
                f'check focus_genes in dataset_configs.py.')
        if len(usable) < len(focus_genes):
            _log(f'NOTE: artifacts cover {usable}; no betadata for '
                 f'{[g for g in focus_genes if g not in set(trained)]}')

        if adata is None:
            adata = _load_adata(paths, cfg['cell_type_src'])
        tiers = [t for t in cfg['tiers'] if t in adata.obs.columns]
        if not tiers:
            raise ValueError(f'none of {cfg["tiers"]} are columns of adata.obs')
        _log(f'artifacts over tiers {tiers} -> {paths["metab_outdir"]}')

        beta_analysis.write_metabolites(
            str(paths['betadata']), adata.obs, tiers, str(paths['metab_outdir']), genes=focus_genes)
        _log('wrote metabolites.csv')

        beta_analysis.write_histograms(
            str(paths['betadata']), adata.obs, tiers, str(paths['metab_outdir']),
            genes=focus_genes, plot=True)
        _log('wrote histograms.csv + histograms.png')

        # Per-cell betas onto the full adata. group=None keeps every modulator group
        # (tf + lr + ltf + metab), which is what makes this file big -- shapes below.
        beta_analysis.betas_to_adata(
            adata, str(paths['betadata']), genes=focus_genes, group=cfg['beta_group'])
        for gene, mods in adata.uns.get('beta_modulators', {}).items():
            _log(f'  beta_{gene}: {adata.obsm[f"beta_{gene}"].shape} ({len(mods)} modulators)')
        _log(f'writing {paths["beta_adata"]}')
        adata.write_h5ad(paths['beta_adata'])
        _log(f'wrote {paths["beta_adata"]} '
             f'({paths["beta_adata"].stat().st_size / 1e9:.1f} GB)')

    _log(f'=== {dataset} done ===')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS),
                        help='dataset folder under PROJECT_DATA_DIR')
    parser.add_argument('--stage', nargs='+', default=list(STAGES), choices=list(STAGES),
                        help='stages to run (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='delete input_data/ and redo setup (betadata is kept); '
                             'requires the setup stage')
    parser.add_argument('--clear-betadata', action='store_true',
                        help='delete betadata/, forcing every gene to retrain '
                             '(independent of --overwrite)')
    parser.add_argument('--data-dir', default=PROJECT_DATA_DIR)
    args = parser.parse_args(argv)

    stages = [s for s in STAGES if s in args.stage]   # always in canonical order
    run_dataset(args.dataset, stages=stages, overwrite=args.overwrite,
                clear_betadata=args.clear_betadata, data_dir=args.data_dir)


if __name__ == '__main__':
    main()
