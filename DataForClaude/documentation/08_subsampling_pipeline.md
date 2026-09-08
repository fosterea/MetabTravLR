# Gene-pair permutation subsampling (SpaceTravLR)

> Built 2026-09-07 from `specs/subsample_spec.md`. Studies how **stable** the learned per-gene-pair
> metabolite coefficients are by re-training SpaceTravLR on many random subsets of a curated
> transporter-pair panel. **No `src/SpaceTravLR/` changes** — everything is in
> `metab_processing/SpaceTravLR/` and rides the existing `SpaceShip.fit(metabolites=...)` path.

## What it does
1. Foster curates a small base panel `sample_metabolites.yaml` (same schema as
   `metabolite_selection.yaml`) next to the other yamls in `harreman_outputs/`.
2. **Sample** (local, cheap): Bernoulli-keep each gene pair with prob `p`; drop metabolites
   left empty; retry a wholly-empty draw. Writes `n_permutations` files to a fresh run.
3. **Fit** (SLURM): train SpaceTravLR on each subsample, then aggregate the metabolite betas.

Key modeling choice (per spec): each **surviving gene pair is its own "metabolite" column**,
`metab@{Metabolite}-{g1}_{g2}` — a single pair (itself), **both orientations summed** for a
heterotypic pair. This is NOT `metab_loader.build_metabolites` (which merges/sums pairs within a
metabolite, D11); it is a separate `pairs_to_metabolites` formatter, so each pair is attributed
to its own metabolite. Deliberate tradeoff: two metabolite names sharing a pair → two collinear
columns in one fit (accepted at the near-OLS defaults; the point is per-pair attribution).

## Two deliverable files (both in `metab_processing/SpaceTravLR/`)
- **`run_subsamples.py`** — importable logic + the SLURM job body (CLI). Functions:
  - `sample_once(selection, p, rng)`, `write_subsamples(dataset, n_permutations, p, data_dir, seed)`
    → next `run_{r}`, returns `r`. `latest_run(dir)`.
  - `pairs_to_metabolites(selection, var_names=None)` → `{f"{name}-{g1}_{g2}": pairs}` for `fit`.
  - `run_subsamples(dataset, run=-1, overwrite=False, cell_type_col=None, data_dir=...)` — the job.
  - `build_run_analysis(...)`, `clear_markers(dataset, run=-1, data_dir=...)`, `main()` CLI.
- **`subsample_permutations.ipynb`** — driver: sample → SLURM dispatch cell (vars: `dataset`,
  `run`, `overwrite`, `cpus` default 6, `cell_type_col`) → clear-markers cell.

## Directory layout
```
<dataset>/easy_download/harreman_outputs/
    sample_metabolites.yaml                              # Foster's curated base panel
    subsamples/run_{r}/sampled_metabolites_{j}.yaml      # the draws
<dataset>/spacetravlr_subsamples/run_{r}/
    _setup/spacetravlr_output/input_data/...             # ONE shared setup (see below)
    subsample_{j}/spacetravlr_output/
        input_data -> symlink to ../../_setup/.../input_data
        betadata/<gene>_betadata.parquet
    subsample_{j}/DONE                                   # completion marker (resume skips these)
    subsample_betas.h5ad                                 # analysis A (self-contained adata)
    subsample_beta_means.csv                             # analysis B (tidy per-cell-type means)
```
(yamls are `.yaml`, matching `metabolite_selection.yaml`.)

## Design decisions (Foster's, 2026-09-07)
- **Shared setup per run.** SpaceTravLR setup (MAGIC/CellOracle/NicheNet) is
  metabolite-independent, so it is built ONCE per run and **symlinked** into each subsample's
  `input_data`; only `fit` re-runs per subsample. This avoids N× the expensive setup. Verified
  safe: `fit` only *reads* `input_data`; it writes betadata + `run_params.json` to each
  subsample's own `betadata/`. Guarded by `_setup_lock` (reused from `run_spacetravlr.py`) — two
  concurrent jobs writing the one `_adata.h5ad`/`celloracle_links.pkl` would corrupt it.
- **`overwrite` = redo the (shared) setup only.** It `rmtree`s `input_data` and rebuilds; it does
  NOT clear `DONE` markers (those subsamples keep their betas, trained on the previous setup — a
  NOTE is logged). Use `clear_markers` to force a re-fit.
- **Annotation column is a CLI variable** (`--cell-type-col`; Alexi UC = coarse
  `25_06_11_ICI_5K_Coarse_annotations`). Falls back to the dataset config's `cell_type_src`.
- **Both analysis objects.** A = `subsample_betas.h5ad`, a **self-contained analysis adata**:
  the display adata (raw counts in `X`, cell-type annotation in `obs`) with, as `obsm`, the
  betas (`beta_{gene}__sample{j}`, read from the betadata parquet DB; JSON `uns['subsample_index']`
  maps key→sample/gene/columns) **and the metabolite communication scores** `x_metab` (cells ×
  gene-pair, names in `uns['x_metab_modulators']`) — computed ONCE per run over the union of drawn
  pairs (`beta_analysis.compute_metab_x`, in the SLURM job where torch lives) so the downstream
  **analysis reads only this file with pure pandas/numpy — no SpaceTravLR/torch, no re-diffusion**.
  `x_metab` is skipped (with a NOTE, betas intact) if the processed adata / `run_params.json` is
  missing. B = `subsample_beta_means.csv` (tidy, via `beta_analysis.tier_means`, `sample` column).
- **Analysis notebook:** `metab_processing/Analysis/Foster/subsample_uc.ipynb` — loads A only;
  average β per gene pair in a cell type, per-pair β distributions, and R² of the cell type's raw
  counts vs β·x (`beta[:,pair]·x_metab[:,pair]`). Cell type starts at `T`.
- **A "0-row" `subsample_beta_means.csv`** means either no `metab@` columns were trained (a
  `sample_metabolites.yaml` gene-symbol vs panel mismatch) OR the trained cells carry no label
  under `cell_type_col` (`tier_means`' `groupby` drops NaN groups). `build_run_analysis` logs a
  distinct WARNING for each; the notebook's Diagnostic pins it down on real data.
- **Resumable, linear.** A `DONE` marker per subsample; a resubmitted job (or a walltime kill)
  picks up the next unmarked subsample. `fit`'s own per-gene parquet resume composes under that.
  Running just the first subsample = sample `n_permutations=1`.

## Usage
```python
# notebook
run = write_subsamples('13473_HS4_UC-Slice_1', n_permutations=1, p=0.5,
                       data_dir=f'{DATA_DIR}/Alexi_UC_Spliced', seed=0)
# then the SLURM dispatch cell -> run_subsamples.py --dataset ... --run -1
#   --cell-type-col 25_06_11_ICI_5K_Coarse_annotations --data-dir {DATA_DIR}/Alexi_UC_Spliced
```

## Tests
`tests/test_subsamples.py` (26, Tier-0, no torch/SpaceShip/harreman): the pure functions,
`build_run_analysis` over fake `beta_metab@...` parquets, and a **mocked `run_subsamples`**
(SpaceShip stubbed) covering shared-setup + symlink + `DONE`-resume + `overwrite` rebuild.
The real SLURM body (torch training) is only validated on Savio.

## Dev/review provenance
Plan → independent **critic** (found the symlink-safety confirmation + YAML-tuple/uns-encoding
fixes) → **metab-dev** implemented File 1 + tests → **metab-review** (Majors: add the setup lock;
add mocked orchestration tests — both applied). Flagged to Foster: folder spelling
`spacetravlr_subsamples` (vs spec's "spacetravler"); the shared-setup deviation from spec's
per-subsample setup.
