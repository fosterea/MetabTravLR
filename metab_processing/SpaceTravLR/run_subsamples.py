#!/usr/bin/env python
"""Gene-pair permutation-subsampling pipeline for MetabTravLR.

Trains SpaceTravLR on random subsets of a curated metabolite panel's transporter gene
pairs, so we can study how stable the learned per-gene-pair coefficients are. Each
surviving gene pair is modeled as its OWN "metabolite" column (both orientations summed
via `pairs_to_metabolites`), i.e. one `beta_metab@{name}-{g1}_{g2}` column per pair.

Two sides, kept in separate directory trees:
  - sampling (cheap, notebook-facing): `<dataset>/easy_download/harreman_outputs/
    subsamples/run_{r}/sampled_metabolites_{j}.yml`, written by `write_subsamples`.
  - training (SLURM job body): `<dataset>/spacetravlr_subsamples/run_{r}/`, with ONE
    shared setup (`_setup/spacetravlr_output/`, since setup is metabolite-independent)
    symlinked into each `subsample_{j}/spacetravlr_output/input_data`, and a `DONE`
    marker per subsample for resume (`clear_markers` deletes these to force a re-fit).

Note: a metabolite NAME containing '$' or '#' would misclassify under
`beta_analysis._group` (which keys on those separators); real names use commas, spaces,
hyphens, parens, and apostrophes, so this is low risk and not guarded against here.

    python run_subsamples.py --dataset Primary_Dermal_Melanoma
    python run_subsamples.py --dataset Human_Lung --run 2 --overwrite
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
import json
import os
import re
import shutil

import anndata
import numpy as np
import pandas as pd
import yaml

from SpaceTravLR.spaceship import SpaceShip
from metab_processing.metab_travlr_config import PROJECT_DATA_DIR
from metab_processing.SpaceTravLR import beta_analysis
from metab_processing.SpaceTravLR.dataset_configs import dataset_paths, get_config
from metab_processing.SpaceTravLR.metab_loader import load_metabolite_selection
from metab_processing.SpaceTravLR.run_spacetravlr import (
    _drop_tiny_clusters,
    _isolate_cache_dir,
    _load_adata,
    _log,
    _processed_var_names,
    _setup_lock,
    _trained_genes,
    setup_is_complete,
)

_RUN_RE = re.compile(r"run_(\d+)$")
_SAMPLE_INDEX_RE = re.compile(r"_(\d+)$")
_SAMPLE_YAML_RE = re.compile(r"_(\d+)\.yml$")


# --------------------------------------------------------------------- sampling (pure)
def sample_once(selection, p, rng):
    """One random subsample of `selection` (`{metabolite: [(g1, g2), ...]}`).

    Keeps each gene pair independently with probability `p` (Bernoulli via
    `rng.random() < p`). A metabolite left with no pairs is dropped entirely. Pure, no
    I/O; `rng` is a `numpy.random.Generator` (or anything exposing `.random()`).
    """
    sampled = {}
    for name, pairs in selection.items():
        kept = [pair for pair in pairs if rng.random() < p]
        if kept:
            sampled[name] = kept
    return sampled


def latest_run(subsamples_dir) -> int:
    """Max existing `run_{i}` number directly under `subsamples_dir`; 0 if none/missing."""
    subsamples_dir = Path(subsamples_dir)
    if not subsamples_dir.is_dir():
        return 0
    runs = [int(m.group(1)) for p in subsamples_dir.iterdir()
            if (m := _RUN_RE.match(p.name))]
    return max(runs, default=0)


def write_subsamples(dataset, n_permutations, p, data_dir, seed=None) -> int:
    """Write `n_permutations` random subsamples of the base curated panel to a new run.

    Reads `<harreman_outputs>/sample_metabolites.yml` (Foster's curated base subset, same
    schema as `metabolite_selection.yaml`), draws `n_permutations` independent subsamples
    via `sample_once`, and writes each as `sampled_metabolites_{j}.yml` (j = 1..n) under a
    fresh `subsamples/run_{r}/` directory (r = one past the highest existing run). A draw
    that comes back completely empty (no pair survives in any metabolite) is retried.
    Returns `r`.
    """
    assert 0 < p <= 1, f"p must be in (0, 1], got {p!r} (p=0 would retry forever)"

    paths = dataset_paths(dataset, data_dir)
    harreman_dir = paths["selection_yaml"].parent
    selection = load_metabolite_selection(harreman_dir / "sample_metabolites.yml")

    subsamples_dir = harreman_dir / "subsamples"
    r = latest_run(subsamples_dir) + 1
    run_dir = subsamples_dir / f"run_{r}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for j in range(1, n_permutations + 1):
        sampled = sample_once(selection, p, rng)
        while not sampled:   # retry an empty draw rather than write a useless file
            sampled = sample_once(selection, p, rng)
        doc = {
            "metabolites": [
                {"name": name, "gene_pairs": [list(pair) for pair in pairs]}
                for name, pairs in sampled.items()
            ]
        }
        with open(run_dir / f"sampled_metabolites_{j}.yml", "w") as f:
            yaml.safe_dump(doc, f)

    _log(f"write_subsamples: {dataset} run_{r}: wrote {n_permutations} files "
         f"(p={p}, seed={seed}) -> {run_dir}")
    return r


# ------------------------------------------------------------- per-pair fit formatter
def pairs_to_metabolites(selection, var_names=None):
    """Expand a sampled `{metabolite: [(g1, g2), ...]}` selection so EVERY surviving gene
    pair becomes its own "metabolite" column: `{f"{name}-{g1}_{g2}": pairs}`.

    For metabolite `name` with pair `(g1, g2)`, the resulting column's pairs are
    `[(g1, g2)]` if homotypic, else `[(g1, g2), (g2, g1)]` (both orientations, summed by
    the estimator like any other metabolite). Pairs are deduped within a metabolite by
    unordered identity, and (if `var_names` is given) dropped when either gene is absent
    from the panel. Order is deterministic (first-seen). The result is ready to pass
    straight to `SpaceShip.fit(metabolites=...)`.

    Deliberate: unlike `metab_loader.build_metabolites` (which MERGES metabolites with an
    identical pair-set to avoid perfectly-collinear duplicate predictors under near-OLS,
    see D11), this keeps ONE column per (metabolite name, pair) so each pair is attributed
    to its own metabolite. Two metabolite names sharing the same gene pair therefore yield
    two bit-identical design columns in one fit -- accepted here as the price of per-pair,
    per-metabolite attribution (the whole point of the subsampling study).
    """
    var_set = set(var_names) if var_names is not None else None
    result = {}
    for name, pairs in selection.items():
        seen = set()
        for g1, g2 in pairs:
            if var_set is not None and (g1 not in var_set or g2 not in var_set):
                continue
            key = frozenset((g1, g2))
            if key in seen:
                continue
            seen.add(key)
            col_name = f"{name}-{g1}_{g2}"
            result[col_name] = [(g1, g2)] if g1 == g2 else [(g1, g2), (g2, g1)]
    return result


# --------------------------------------------------------------------------- SLURM job
def clear_markers(dataset, run=-1, data_dir=PROJECT_DATA_DIR) -> int:
    """Delete every `subsample_{j}/DONE` marker under `spacetravlr_subsamples/run_{r}/`,
    forcing the next `run_subsamples` call to re-fit them. Returns the count removed.
    """
    paths = dataset_paths(dataset, data_dir)
    r = latest_run(paths["selection_yaml"].parent / "subsamples") if run == -1 else run
    sub_root = paths["dataset_dir"] / "spacetravlr_subsamples" / f"run_{r}"
    removed = 0
    for done in sub_root.glob("subsample_*/DONE"):
        done.unlink()
        removed += 1
    _log(f"clear_markers: {dataset} run_{r}: removed {removed} DONE marker(s)")
    return removed


def run_subsamples(dataset, run=-1, overwrite=False, cell_type_col=None,
                    data_dir=PROJECT_DATA_DIR):
    """The SLURM job body: fit every sampled subsample in a run, then build the analysis
    objects. One SHARED setup (metabolite-independent) is built once and symlinked into
    each `subsample_{j}/spacetravlr_output/input_data`; already-`DONE` subsamples are
    skipped (resume). See the module docstring for the directory layout.
    """
    cfg = get_config(dataset)
    paths = dataset_paths(dataset, data_dir)
    cell_type_src = cell_type_col or cfg["cell_type_src"]
    focus_genes = cfg["focus_genes"]

    subsamples_dir = paths["selection_yaml"].parent / "subsamples"
    r = latest_run(subsamples_dir) if run == -1 else run
    run_yaml_dir = subsamples_dir / f"run_{r}"
    sampled_paths = sorted(
        run_yaml_dir.glob("sampled_metabolites_*.yml"),
        key=lambda p: int(_SAMPLE_YAML_RE.search(p.name).group(1)),
    )
    if not sampled_paths:
        raise FileNotFoundError(f"no sampled_metabolites_*.yml under {run_yaml_dir}")

    _isolate_cache_dir()

    sub_root = paths["dataset_dir"] / "spacetravlr_subsamples" / f"run_{r}"
    shared_setup_out = sub_root / "_setup" / "spacetravlr_output"
    shared_input = shared_setup_out / "input_data"
    shared_paths = {"adata": paths["adata"], "input_data": shared_input}

    # ------------------------------------------------------------- shared setup
    if overwrite and shared_input.exists():
        _log(f"overwrite: removing {shared_input}")
        shutil.rmtree(shared_input)
        done_subs = list(sub_root.glob("subsample_*/DONE"))
        if done_subs:
            _log(f"NOTE: keeping {len(done_subs)} DONE-marked subsample(s) -- they were fit "
                 f"against the PREVIOUS shared setup. Call clear_markers to re-fit them.")
    if setup_is_complete(shared_setup_out):
        _log(f"run_{r}: shared setup already complete, skipping")
    else:
        # Lock the shared setup: two jobs writing _adata.h5ad / celloracle_links.pkl to the
        # same paths at once would corrupt the ONE setup every subsample symlinks into (HDF5
        # has no concurrent-write support). Same guard run_spacetravlr.py uses.
        with _setup_lock(shared_setup_out):
            adata = _load_adata(shared_paths, cell_type_src)
            adata = _drop_tiny_clusters(adata, cell_type_src)
            ship = SpaceShip(name=dataset.replace("/", "_"), outdir=str(shared_setup_out),
                              genes=focus_genes)
            _log(f"run_{r}: shared setup_ (run_commot={cfg['run_commot']}) ...")
            ship.setup_(adata, overwrite=True, run_commot=cfg["run_commot"])
            if not setup_is_complete(shared_setup_out):
                raise RuntimeError(f"shared setup for {dataset} run_{r} did not complete")
        _log(f"run_{r}: shared setup complete")
    var_names = _processed_var_names(shared_paths)

    # -------------------------------------------------------------- per-subsample fit
    for yaml_path in sampled_paths:
        j = int(_SAMPLE_YAML_RE.search(yaml_path.name).group(1))
        sub_dir = sub_root / f"subsample_{j}"
        sub_out = sub_dir / "spacetravlr_output"
        done = sub_dir / "DONE"
        if done.exists():
            _log(f"run_{r}/subsample_{j}: already done, skipping")
            continue

        sub_out.mkdir(parents=True, exist_ok=True)
        link = sub_out / "input_data"
        if not link.exists():
            os.symlink(shared_input, link)

        selection_j = load_metabolite_selection(yaml_path)
        metabolites = pairs_to_metabolites(selection_j, var_names=var_names)
        n_sampled = sum(len(pairs) for pairs in selection_j.values())
        ship = SpaceShip(name=dataset.replace("/", "_"), outdir=str(sub_out),
                          genes=focus_genes)
        _log(f"run_{r}/subsample_{j}: {n_sampled} sampled pairs -> {len(metabolites)} "
             f"pair-columns (post var-filter) over {len(focus_genes)} target genes")
        ship.fit(metabolites=metabolites, **cfg["fit_kwargs"])
        done.write_text("done\n")
        trained = _trained_genes({"betadata": sub_out / "betadata"})
        _log(f"run_{r}/subsample_{j}: done ({len(trained)} genes with betadata)")

    build_run_analysis(dataset, r, cell_type_src, data_dir)


# ---------------------------------------------------------------------------- analysis
def build_run_analysis(dataset, run, cell_type_col, data_dir=PROJECT_DATA_DIR) -> None:
    """Write both analysis objects for `spacetravlr_subsamples/run_{run}/`:

    - `subsample_betas.h5ad` (Object A, per-cell): one `obsm['beta_{gene}__sample{j}']`
      matrix per (subsample, focus gene) present, plus a JSON-decodable
      `uns['subsample_index']` describing each key (`key`, `sample`, `gene`, `columns`).
    - `subsample_beta_means.csv` (Object B, tidy): `tier_means` per subsample, concatenated
      with a leading `sample` column. Columns: sample, gene, cell_type, modulator, mean,
      std, n.
    """
    paths = dataset_paths(dataset, data_dir)
    cfg = get_config(dataset)
    focus_genes = cfg["focus_genes"]
    sub_root = paths["dataset_dir"] / "spacetravlr_subsamples" / f"run_{run}"

    # We only need the cell labels here, so read obs backed (X stays on disk) rather than
    # loading the whole Xenium adata into memory. `cell_type_col` is an annotation column on
    # the raw adata; `tier_means`/reindex group cells by it.
    backed = anndata.read_h5ad(paths["adata"], backed="r")
    try:
        obs = backed.obs.copy()
        obs_names = backed.obs_names.copy()
    finally:
        if backed.isbacked:
            backed.file.close()
    if cell_type_col not in obs.columns:
        raise KeyError(f"cell_type_col {cell_type_col!r} not in adata.obs of {paths['adata']}")

    sub_dirs = sorted(
        (p for p in sub_root.glob("subsample_*") if _SAMPLE_INDEX_RE.search(p.name)),
        key=lambda p: int(_SAMPLE_INDEX_RE.search(p.name).group(1)),
    )

    # --- Object A: per-cell AnnData
    ra = anndata.AnnData(obs=pd.DataFrame(index=obs_names))
    index_list = []
    for sub_dir in sub_dirs:
        j = int(_SAMPLE_INDEX_RE.search(sub_dir.name).group(1))
        betadata_dir = sub_dir / "spacetravlr_output" / "betadata"
        if not betadata_dir.is_dir():
            continue
        for gene in focus_genes:
            parquet = betadata_dir / f"{gene}_betadata.parquet"
            if not parquet.is_file():
                continue
            mat = beta_analysis._read_betas(parquet, group="metab").reindex(obs_names)
            key = f"beta_{gene}__sample{j}"
            ra.obsm[key] = mat.to_numpy()
            index_list.append({"key": key, "sample": j, "gene": gene,
                                "columns": list(mat.columns)})
    ra.uns["subsample_index"] = json.dumps(index_list)
    ra.uns["run"] = run
    ra.write_h5ad(sub_root / "subsample_betas.h5ad")
    _log(f"run_{run}: wrote subsample_betas.h5ad ({len(index_list)} obsm matrices)")

    # --- Object B: tidy per-cell-type means
    frames = []
    for sub_dir in sub_dirs:
        j = int(_SAMPLE_INDEX_RE.search(sub_dir.name).group(1))
        betadata_dir = sub_dir / "spacetravlr_output" / "betadata"
        if not betadata_dir.is_dir():
            continue
        df = beta_analysis.tier_means(betadata_dir, obs, tier=cell_type_col, group="metab")
        df.insert(0, "sample", j)
        frames.append(df)
    means = (pd.concat(frames, ignore_index=True) if frames else
             pd.DataFrame(columns=["sample", "gene", "cell_type", "modulator", "mean", "std", "n"]))
    means.to_csv(sub_root / "subsample_beta_means.csv", index=False)
    _log(f"run_{run}: wrote subsample_beta_means.csv ({len(means)} rows)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="dataset folder under PROJECT_DATA_DIR")
    parser.add_argument("--run", type=int, default=-1,
                        help="run number to fit (default: -1, the latest sampled run)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the shared setup's input_data/ and redo it")
    parser.add_argument("--cell-type-col", default=None,
                        help="overrides the dataset config's cell_type_src")
    parser.add_argument("--data-dir", default=PROJECT_DATA_DIR)
    args = parser.parse_args(argv)

    run_subsamples(args.dataset, run=args.run, overwrite=args.overwrite,
                    cell_type_col=args.cell_type_col, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
