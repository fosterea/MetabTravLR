"""Submit `run_spacetravlr.py` to SLURM, one job per dataset.

Logs go to ``{METAB_DATA_DIR}/spacetravlr_logs/<DATASET>/`` -- a sibling of
``harreman_logs/``, deliberately outside the dataset's ``spacetravlr_output/``. SLURM
opens the ``--output`` file *before* the job body runs, so its parent directory has to
exist at submit time; ``spacetravlr_output/logs/`` (where ``SpaceShip.spawn_worker``
puts them) does not exist yet on a fresh dataset, which is why setup-over-SLURM failed
instantly there.

    from metab_processing.SpaceTravLR.submit_spacetravlr import submit
    submit('Primary_Dermal_Melanoma')
    submit('Human_Lung', overwrite=True, time_hours=24)
    submit('Human_Lung', stages=['artifacts'], partition='savio3', gres=None)
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


def submit(dataset, stages=None, overwrite=False, clear_betadata=False,
           data_dir=PROJECT_DATA_DIR, dry_run=False, **slurm_overrides):
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
    slurm_cfg = {**cfg['slurm'], **slurm_overrides}

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

    print(f'dataset : {dataset}')
    print(f'stages  : {tag}'
          f'{" (overwrite)" if overwrite else ""}'
          f'{" (clear-betadata)" if clear_betadata else ""}')
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
