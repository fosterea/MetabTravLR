"""Submit `run_spacetravlr.py` to SLURM, one job per dataset.

Logs go to ``{METAB_DATA_DIR}/spacetravlr_logs/<DATASET>/`` -- a sibling of
``harreman_logs/``, deliberately outside the dataset's ``spacetravlr_output/``. SLURM
opens the ``--output`` file *before* the job body runs, so its parent directory has to
exist at submit time; ``spacetravlr_output/logs/`` (where ``SpaceShip.spawn_worker``
puts them) does not exist yet on a fresh dataset, which is why setup-over-SLURM failed
instantly there.

    from metab_processing.SpaceTravLR.submit_spacetravlr import submit, submit_split
    submit_split('Human_Lung')                  # CPU big-mem setup -> GPU fit, chained
    submit('Primary_Dermal_Melanoma')           # one GPU job, all three stages
    submit('Human_Lung', stages=['artifacts'], gres=None)
"""
from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

# Make the repo root importable regardless of CWD or machine.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parent,
)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from metab_processing.metab_travlr_config import PROJECT_DATA_DIR
from metab_processing.SpaceTravLR.dataset_configs import dataset_paths, get_config

RUN_SCRIPT = Path(__file__).resolve().parent / 'run_spacetravlr.py'


def build_command(dataset, stages=None, overwrite=False, clear_betadata=False,
                  data_dir=PROJECT_DATA_DIR, python_path='python') -> str:
    """The `python run_spacetravlr.py ...` line SLURM will execute."""
    cmd = [python_path, str(RUN_SCRIPT), '--dataset', dataset]
    if stages:
        cmd += ['--stage', *stages]
    if overwrite:
        cmd.append('--overwrite')
    if clear_betadata:
        cmd.append('--clear-betadata')
    if str(data_dir) != PROJECT_DATA_DIR:
        cmd += ['--data-dir', str(data_dir)]
    return ' '.join(cmd)


def _slurm_profile(cfg, stages):
    """Which resource block to use: the CPU big-mem one for a setup-only job, else the GPU one.

    Setup uses no GPU (magic + sklearn ridge) but needs far more RAM than an 8-core A40
    allocation provides at Xenium scale, so the two halves want different hardware.
    """
    if stages is not None and list(stages) == ['setup']:
        return {**cfg['slurm'], **cfg['setup_slurm']}, 'setup (CPU big-mem)'
    return dict(cfg['slurm']), 'gpu'


def submit(dataset, stages=None, overwrite=False, clear_betadata=False,
           data_dir=PROJECT_DATA_DIR, dry_run=False, dependency=None, **slurm_overrides):
    """Submit one dataset as a SLURM job. Returns the job id (None if `dry_run`).

    Parameters
    ----------
    dataset : str
        Key in `dataset_configs.DATASETS`.
    stages : list[str], optional
        Subset of ('setup', 'fit', 'artifacts'); None runs all three.
    overwrite : bool
        Delete `input_data/` and redo setup. Trained betadata is kept.
    clear_betadata : bool
        Also delete `betadata/`, forcing every gene to retrain.
    dry_run : bool
        Print the sbatch settings and command without submitting.
    dependency : dict, optional
        Passed to sbatch, e.g. `dict(afterok=<job id>)` to wait on another job.
    **slurm_overrides
        Override any key of the dataset's `slurm` config for this submission
        (`time_hours`, `partition`, `qos`, `gres`, `cpus_per_task`, `account`,
        `job_name`, `python_path`). Pass `gres=None` to drop the GPU request.
    """
    cfg = get_config(dataset)
    paths = dataset_paths(dataset, data_dir)

    unknown = set(slurm_overrides) - set(cfg['slurm'])
    if unknown:
        raise KeyError(f'unknown slurm option(s) {sorted(unknown)}; '
                       f'valid: {sorted(cfg["slurm"])}')
    profile, profile_name = _slurm_profile(cfg, stages)
    slurm_cfg = {**profile, **slurm_overrides}

    stamp = time.strftime('%Y%m%d_%H%M%S')
    tag = '-'.join(stages) if stages else 'all'
    outlog = paths['log_dir'] / f'{tag}_{stamp}.log'

    command = build_command(
        dataset, stages=stages, overwrite=overwrite, clear_betadata=clear_betadata,
        data_dir=data_dir, python_path=slurm_cfg['python_path'])

    sbatch_kwargs = dict(
        account=slurm_cfg['account'],
        partition=slurm_cfg['partition'],
        qos=slurm_cfg['qos'],
        cpus_per_task=slurm_cfg['cpus_per_task'],
        ignore_pbs=True,
        job_name=f'{slurm_cfg["job_name"]}_{dataset}',
        output=str(outlog),
        time=timedelta(hours=slurm_cfg['time_hours']),
    )
    if slurm_cfg['gres']:
        sbatch_kwargs['gres'] = slurm_cfg['gres']
    if dependency:
        sbatch_kwargs['dependency'] = dependency

    print(f'dataset : {dataset}')
    print(f'stages  : {tag}'
          f'{" (overwrite)" if overwrite else ""}'
          f'{" (clear-betadata)" if clear_betadata else ""}')
    print(f'profile : {profile_name}')
    print(f'log     : {outlog}')
    print(f'sbatch  : {sbatch_kwargs}')
    print(f'command : {command}')
    if dry_run:
        print('dry run -- not submitted')
        return None

    # Must exist before sbatch, or the job dies opening its log file.
    paths['log_dir'].mkdir(parents=True, exist_ok=True)

    from simple_slurm import Slurm  # imported late so dry_run works off-cluster

    job_id = Slurm(**sbatch_kwargs).sbatch(command)
    print(f'submitted job {job_id}')
    return job_id


def submit_split(dataset, overwrite=False, clear_betadata=False, data_dir=PROJECT_DATA_DIR,
                 dry_run=False, setup=None, run=None):
    """Submit setup and fit+artifacts as two chained jobs. Returns `(setup_id, run_id)`.

    Both are queued immediately; the second waits on `afterok` of the first, so it accrues
    queue priority while setup runs and never starts if setup fails. This is the route to use
    at Xenium scale: setup gets a CPU big-mem node (it OOMs an 8-core A40 allocation) and only
    the training half holds a GPU.

    `overwrite` applies to the setup job (it is what redoes setup); `clear_betadata` applies
    to the training job. `setup=` / `run=` are dicts of per-job SLURM overrides, e.g.
    `submit_split('Human_Lung', setup={'time_hours': 12}, run={'time_hours': 36})`.
    """
    setup_id = submit(dataset, stages=['setup'], overwrite=overwrite, data_dir=data_dir,
                      dry_run=dry_run, **(setup or {}))
    print()
    dependency = dict(afterok=setup_id) if setup_id is not None else None
    if dry_run:
        print('(the training job would wait on --dependency=afterok:<setup job id>)')
    run_id = submit(dataset, stages=['fit', 'artifacts'], clear_betadata=clear_betadata,
                    data_dir=data_dir, dry_run=dry_run, dependency=dependency, **(run or {}))
    return setup_id, run_id
