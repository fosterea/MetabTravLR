# Decisions & project state (living log)

> The transportable replacement for Claude's private memory. Anything durable we'd want a
> future session (or a teammate) to know goes here. Append-only-ish; mark superseded items.
> Convert relative dates to absolute.

## Who / how we work
- User goes by **Foster** (account/email is `andrew@bankofangus.com`, home dir `fosterangus`;
  address them as Foster).
- Foster wants: an accurate **mental model before code**; **minimal, surgical** edits to the
  existing package with project logic in **separate scripts**; **few dependencies**; docs kept
  in-repo (transportable), not in Claude memory. Don't touch the Sphinx `docs/`.
- Compute: big data (100k–1M-cell Xenium, ~5k-gene panel) runs on the **Savio** cluster; this
  local repo is for development + small-data tests.

## Goal (one line)
Add harreman metabolite transporter pairs as spatial modulators in SpaceTravLR, train (only
the genes of interest), and read the learned `beta_<export>$<import>` coefficients over
labeled gene sets to rank metabolites by effect (e.g. ↑T-cell activity / ↓exhaustion).

## Decisions made
| # | Date | Decision | Notes |
|---|------|----------|-------|
| D1 | 2026-07-10 | **Analyze coefficients directly; no perturbation** (`GeneFactory`/`perturb` out of scope). | Drives the "train only genes of interest" strategy. |
| D2 | 2026-07-10 | Metabolite ≈ L–R edge: **export gene = ligand (diffused), import gene = receptor (local)**. | Reuses existing `extra_lr` machinery. |
| D3 | 2026-07-10 | **Metabolite orientation: start with BOTH directions** (add g1→g2 and g2→g1); let group-lasso prune. | Revisit harreman sender→receiver orientation later if needed. |
| D4 | 2026-07-10 | Keep durable notes in `DataForClaude/documentation/` + a root `CLAUDE.md`; **do not use Claude private memory**. | This file replaces the old memory entries. |
| D5 | 2026-07-10 | Restrict training to **target genes of interest** (valid because we read β directly). | Seed `OracleQueue` with a gene subset; biggest tractability win. |
| D6 | 2026-07-10 | Metabolites are their **own new modulator group** (a 5th group), **not** folded into the L–R `extra_lr` group. | Separate group-lasso group + **distinct betadata separator** (e.g. `@`, since `$`/`#` are taken) so metabolite β's are independently identifiable. See `01_pipeline_deep_dive.md`. |
| D7 | 2026-07-10 | **Keep TF + L–R modulators, but optional** (default keep). | Metabolite β estimated controlling for known regulation. Likely **skip COMMOT** (harreman is our prior). |
| D8 | 2026-07-16 | **Fix the received-ligand kernel: hard cutoff at `radius` + narrow σ; row-chunk.** Committed `793c096`. | Auto mode, **verified against the paper's actual methods text** (Foster's prompt to check). Finding: the paper *mandates a hard cutoff* — "spatial neighbors n as all locations i within a circle … with a predefined radius r", summing over only in-radius neighbors — so the old "fast" kernel (`σ=radius`, **no cutoff**, sum over all cells) genuinely **violates the paper** (real bug; OOM'd at 57 GB @100k). The paper does **NOT** specify σ (defers to CytoSignal); `σ=radius/3.72` is the codebase's own `gaussian_kernel_2d` convention (makes the Gaussian ≈0 at the cutoff) and is what the original slow `compute_radius_weights` uses. So the fix = align fast path to `gaussian_kernel_2d`: cutoff = paper-mandated, σ = code-convention. → ~1.34 GB @20k, matches narrow reference to ~2e-16, chunk-invariant. **CHANGES results vs old wide kernel — intended.** Old wide fns kept as `*_wide_deprecated`. Paper refs: Methods "Spatially informed signaling inference" (p17); σ nuance in `05`/agent trace. |
| D9 | 2026-07-20 | **Harreman aggregate CCC OOM → gene-pair chunking (bit-identical), NOT lowering M.** The `HarremanRunner` OOM at Xenium scale is in `compute_{,ct_}cell_communication`'s dense `(n_cells × n_gp)` matmul intermediates — a *different* problem from the per-cell §5 OOM (already fixed via `nbhd_scores`→`compute_interacting_cell_scores_lowmem`). Their permutation null is already cell-reduced, so `M` is irrelevant to their memory. | Chunk the **gene-pair** axis (provably bit-identical: score sums over cells; column slicing + per-row DANB standardize don't reorder). Simplest fix that solves it at any scale; "chunk matmul only" was rejected once we verified `standardize_counts` is per-row (→ chunk `counts_1/2` too). Adaptive default chunk `= max(1, 50M//n_cells)`. Reproduced two stock float32 quirks for exactness. See `05` §5c; drop-ins in `cell_communication_lowmem.py` (CU-A–D). |
| D10 | 2026-07-21 | **Per-cell nbhd OOM (≥600k cells) → Option B: two-pass gene-pair + metabolite chunking, preserve the exact `uns` contract.** Foster chose **B over A** (stream-to-summary): our wall is **GPU** memory, not RAM, and B is simpler (no `summarize_nbhd_scores` refactor). B bounds GPU to `(n_cells, chunk)` but still stores the full `(n_cells, n_gp)`/`(n_cells, n_m)` matrices on **CPU** — accepted. | CU-E in `compute_interacting_cell_scores_lowmem`'s `np` branch. Params `gene_pair_chunk_size`/`metabolite_chunk_size` threaded through `nbhd_scores.compute_nbhd_scores` **only** (not `HarremanRunner`). Metabolite pass recomputes union gene-pair scores (~2× perm matmuls, sanctioned). **Review-caught bug:** observed `cs_m` must be reduced on the **same device** (GPU) as the perm scores — a CPU-side `.sum(dim=1)` over ≥3 pairs can ULP-differ from CUDA → flip an exceedance; CPU tests can't see it. See `07` §10, `05` §5. |

## Leaning / proposed (not final)
- Signed gene-set score: `mean_{positive} β̄ − mean_{negative/exhaustion} β̄`.

## Open decisions (see `02_metab_integration_notes.md` §4)
- Keep vs. optional TF/LR (default); handling target genes with no CellOracle TF links
  (the no-TF **orphan-skip** in `oracles.py:475`).
- Gene-set input format (JSON vs dict) — first labels: `positive` (activity) / `negative`
  (exhaustion).
- Which harreman tier drives edge selection; whether to restrict edges by sender/receiver
  cell type.

## Key technical facts (so we don't re-derive)
- Repo renamed **SpaceOracle → SpaceTravLR** (both names in code/paths).
- Two phases: **train** (`SpaceShip.setup_` → `run_spacetravlr` → `betadata/{gene}_betadata.parquet`)
  vs **simulate** (`GeneFactory.perturb`, out of scope).
- Coefficients: betadata columns `beta0`, `beta_<TF>`, `beta_<lig>$<rec>` (L–R ← our
  metabolites), `beta_<lig>#<TF>` (L–TF). Produced by `CellularNicheNetwork` as
  `sigmoid(MLP(spatial))·anchors` (anchors = group-lasso solution).
- CellOracle runs **once** at setup (`spaceship.run_celloracle_`, vendored `celloracle_tmp`,
  base GRN `SpaceTravLR_data/{species}_base_grn.parquet`) → `celloracle_links.pkl`; at
  train time only the pickle is read.
- **Tractability blockers:** dense **O(N²)** ops (`_gaussian_kernel_2d_batch`,
  `create_spatial_features` `cdist`) OOM at ≥100k cells. See `02_...md` §3 for fixes.
- Harreman data contract & bugs: see `02_...md` §1. `extra_lr` hook: `models/parallel_estimators.py`.

## Known pre-existing bugs (found during Run 1)
Confirmed present at HEAD, independently reproduced. Relevant because our metabolite work
exercises the same paths.
- ✅ **FIXED — `activation` kwarg crash** — `parallel_estimators.py` fallback branch passed
  `activation=self.activation` into `CellularNicheNetwork.__init__`, which doesn't accept it →
  crashed any real `fit()` hitting the group-lasso **R²<0.15 "poor fit" branch**. Reproduced on
  real data. Fixed by dropping the dead kwarg (identity behavior unchanged) in commit `2699947`;
  the gene-focus end-to-end test now runs unpatched as a regression test.
- **`get_betas` non-contiguous `cell_type_int`** — NOT a real bug. `process_adata_` creates
  `cell_type_int` contiguously (`encode_labels`, `0..n-1`); the failure only appeared in a
  hand-subset test. No fix needed (tests remap to contiguous, as real setup does).
- **pyarrow extension-type flake** — `run_celloracle_` tests corrupt pyarrow's registry
  (`ArrowKeyError: pandas.period already defined`), breaking later real parquet I/O in the same
  process. Order-dependent test flake; our tests sidestep it by mocking parquet I/O. Deferred.
- **`spawn_worker(clusters=…)`** — `test_spacetravlr.py::test_spawn_worker` fails at HEAD
  (signature changed for Savio, test not updated). **Paused per Foster** (real logic is what
  matters); left as a known pre-existing red.
- ✅ **FIXED — pyarrow suite flake** — tests mocking `sys.modules`/`importlib` (the celloracle
  tests) made pyarrow re-register pandas extension types → `ArrowKeyError: pandas.period already
  defined` on later real parquet I/O. Fixed with an idempotent-registration shim in
  `tests/conftest.py` (commit `4071df8`). Test-only artifact; never affected the real pipeline.
- **`plotting/niche.py:34`** — calls `received_ligands(xy=…, lig_df=…, radius=…)` with keyword
  names that don't match the function signature (`ligands_df`, `lr_info`, no `radius`). Broken at
  HEAD *before* CU-5 (our change only added an additive `eps` kwarg). Pre-existing; not touched.

## Setup-cost complexity (from the "don't skip other genes" investigation, 2026-07-11)
- **CellChat (L–R) + NicheNet (L–TF) modulator construction are ALREADY skipped for non-focus
  genes** — they run inside the per-gene estimator (`init_ligands_and_receptors`,
  `grn.get_regulators`), only built for queued (focus) genes. No slowdown; no CellChat-vs-NicheNet
  inconsistency (same per-gene mechanism).
- **The only all-gene setup cost is the CellOracle base-GRN build** (`run_celloracle_`):
  `get_links` fits bagging-ridge **per cluster × per target gene** over the base GRN's targets →
  `O(#clusters × #targets × 20 × ridge(#cells × #TFs))`, **linear in the full gene count**. At
  100k–1M cells × ~5k-gene panels this is real and avoidable. **Plan (greenlit): filter the base
  GRN to `gene_short_name ∈ focus_genes` before `import_TF_data`** so `get_links` only fits focus
  targets — lossless for us (we only need a focus gene's own TF regulators; we don't propagate).
  → dev/review loop, upcoming.
- **The real scaling wall is the dense O(N²) received-ligand kernel + `cdist` spatial features**
  (cell-count-driven, `02_...md` §3 / CU-5) — orthogonal to gene focus; gene focus won't help it.
- **Full O(N²) / memory audit for CU-5** (2026-07-11): (a) `_gaussian_kernel_2d_batch` (received
  ligands) — dense N×N; (b) `create_spatial_features` `cdist` — dense N×N; (c) `xyc2spatial_fast`
  spatial maps — *linear* in N but materializes the whole `(N, n_clusters, 64, 64)` tensor (~160 GB
  @1M) → must be batched/streamed too; (d) **MAGIC imputation** — Foster reports it runs tractably
  at 100k, so deprioritized (verify backend only if we push toward 1M). CU-5 = (a)+(b) sparse
  radius-neighbors + (c) batched spatial maps.

## ✅ Real-data full-pass validation (2026-07-11)
`scripts/real_data_smoke.py` ran the whole `SpaceShip` pipeline on real data
(`snrna_germinal_center.h5ad`, 1309 cells) with `genes=['CD74','BCL6','FOXO1']`:
- `setup_` 73s (MAGIC + CellOracle + NicheNet download all worked); `run_spacetravlr` 64s.
- **CellOracle restriction verified live:** links covered exactly the 3 focus targets, no others.
- **Real CNN training exercised** (R² 0.72–0.99 — not the fallback shortcut), full modulator
  assembly (e.g. CD74 = 20 TF + 687 L–R + 96 L–TF). Activation fix let fallback branches run.
- Only the 3 focus genes trained; `BCL6_betadata.parquet` = 1309×740, all-finite, 239k nonzero.
This validates gene-focus + the CellOracle restriction + the activation fix **together**, and
closes the "tests only hit the R²<0.15 shortcut" gap.

## Change-unit status
- **CU-3 (gene focus, `SpaceShip(genes=…)`):** ✅ done + **committed** `2e523aa`, 2026-07-11.
  Restricts the training `OracleQueue` to the focus genes; does not subset `adata.var` or the
  GRN/received-ligand setup. Guards: bare-string `TypeError`, empty-list/missing-gene
  `ValueError`, dedupe. Tests: `tests/test_gene_focus.py` (10, incl. a multi-gene "attempts/writes
  exactly the focus set" proof + a real-data end-to-end) + 5 plumbing tests in
  `tests/test_spacetravlr.py`. Foster's jscatter changes stayed uncommitted (hunks isolated).
- **Activation-bug fix:** ✅ committed `2699947`.
- **CellOracle base-GRN focus restriction (perf):** greenlit; next via dev/review loop. Filter
  base GRN to focus targets in `SpaceShip.run_celloracle_` when `self.focus_genes` is set.
- **CU-5 (O(N²) sparse):** ✅ committed `691ab41`, 2026-07-11. Behavior-preserving:
  `create_spatial_features` → cKDTree radius-count (**bit-exact**); received-ligand kernel →
  sparse cKDTree (cutoff `C=r·√(2·ln(1/eps))`, `eps=1e-9`; measured max **relative** error vs
  dense ~4.7e-7, within rtol=1e-6; self-pair preserved). Dense kept as `*_dense` references;
  18 equivalence tests (`tests/test_sparse_equivalence.py`); Opus-reviewed, no blockers; real-data
  smoke passes on the sparse path. **Unblocks 100k.**
  Follow-ups (deferred): (a) at **1M cells** the large secreted radius can still make the
  in-cutoff neighbor set big (~hundreds of GB) → row-chunk the KDTree query; (b) `xyc2spatial_fast`
  spatial-map tensor (~160 GB @1M) → batch/stream it. Both needed only for 1M-scale, not 100k.
- **CU-5b (received-ligand OOM at Xenium scale, D8):** ✅ committed `793c096`, 2026-07-16.
  Diagnosis: at Xenium density (~15 µm spacing) with `radius=300`, the received-ligand matrix hit
  **57 GB @100k / 202 GB @300k** (crashed Foster's in-kernel `fit()` on the melanoma Xenium set).
  Fix (D8) = narrow kernel + hard cutoff at `radius` + row-chunking → ~1.34 GB @20k, bounded.
  `create_spatial_features` (1.4 GB @100k) is fine. **NEXT ceiling = the spatial-maps tensor**
  (`xyc2spatial_fast`, ~20 GB @100k, 59 GB @300k) — batch/stream it (CU-5c). Needed for 300k+;
  100k should now fit on a normal node.
- **Gene-focus missing-gene handling:** now **drops missing focus genes with a printed warning**
  (was ValueError); all-missing still errors. Committed `fd815ed`.
- **get_betas GPU OOM (112k Xenium run, 2026-07-17):** the actual crash on Foster's 112,551-cell
  melanoma run (10.5 GB GPU) was a CUDA OOM in `get_betas` — it ran the CNN forward on a whole
  cluster's cells at once (unbatched). ✅ Fixed `63f0400`: **batch it (4096) in eval mode**. `get_betas`
  used train-mode BatchNorm (full-cluster stats), inconsistent with `predict()` (eval); measured
  eval-vs-train β diff is **max abs 1.19e-7 / mean 5.6e-10** (negligible), so eval-mode batching
  is effectively behavior-preserving and correct. **Full data-flow / memory analysis + efficiency
  roadmap + the metabolite-only and CNN-scaling answers are in `06_efficiency_and_dataflow.md`.**
  Next ceiling after this = the spatial-maps tensor (~20 GB@100k, CU-5c).
- **CU-1 (metabolite group):** ✅ done + **committed** `1602172`, 2026-07-17. `metab_pairs` arg on
  `SpatialCellularProgramsEstimator` → group-lasso group #5, `beta_<export>@<import>` columns via a
  `metabolite_interactions` static method (`received_ligand_tfl(export, diffused) × import(local)`,
  mirrors the L–TF path). Column order kept consistent across `train_df`/`self.modulators`/groups/
  betadata names. Diffusion cache made target-agnostic (`_diffusion_extra_lr`) so the first gene's
  shared `received_ligands_tfl` holds every export gene later genes need. Guards: target-gene
  exclusion, drop genes absent from `var_names`, dedup, fail-fast type validation. Default None →
  byte-identical. Opus-reviewed (no blockers). Tests `tests/test_metab_group.py` (group-5 pinned,
  known-answer flux incl. duplicate transporter genes, real-diffusion path).
- **Metabolite loader:** ✅ `metab_processing/SpaceTravLR/metab_loader.py` (committed `1602172`). `metabolite_
  selection.yaml` → `{metabolite: [(g1,g2)]}` (grouped, for read-back) + flat deduped `metab_pairs`
  (homotypic once, heterotypic **both orientations** per D3, optional `var_names` filter). The real
  file = 76 metabolites → 144 model pairs. `tests/test_metab_loader.py`.
- **CU-2 (thread `metab_pairs` + relax orphan skip):** ✅ done + **committed** `68ec51d`, 2026-07-17.
  `oracles.SpaceTravLR(metab_pairs=…)` → estimator; `SpaceShip.run_spacetravlr(metab_pairs=…)`
  forwards it (`fit(metab_pairs=…)` works). Orphan gate #1 relaxed: orphan iff no TF regulators AND
  `n_metab==0` (byte-identical when metab absent). Gate #2 now decides write-vs-orphan on the
  **post-zero-filter** column count, so a metab-only gene whose fit hits the R²<0.15 zeroed-anchor
  fallback is orphaned cleanly (was: degenerate 0-column parquet). Reviewed (Medium fixes applied).
  Tests `tests/test_metab_wiring.py` incl. a REAL-estimator Tier-1 driving a TF-less metab-only gene
  through `run()` to a finite `beta_@` column.
- **Beta-analysis read-back:** ✅ `metab_processing/SpaceTravLR/beta_analysis.py` (rewritten
  2026-07-19 — see that session's entry below). `tests/test_beta_analysis.py`.
- **Quickstart notebook:** ✅ `metab_processing/quick_start_metab.ipynb` (committed `c04ccad`) —
  data-dir/dataset config, editable gene-sets cell, yaml→`metab_pairs`, `setup_`(COMMOT off)+
  focus-gene `fit(metab_pairs=…)`, coefficient read-back; results under the dataset dir.

## Key finding — regularization is near-zero, so "direction from which orientation survives" ≠ pruning (2026-07-17)
`SpatialCellularProgramsEstimator.fit` defaults: `group_reg=threshold_lambda=1e-7`, `l1_reg=1e-9`
(`parallel_estimators.py:1146-1147`) → **effectively near-OLS**; groups are essentially never zeroed
and there is **no within-group sparsity**. Consequences for the metabolite work (Foster's per-pair,
both-orientations plan): both orientations of a heterotypic pair get nonzero β's, so **direction is
read from β *magnitude*** (|export@import| vs |import@export| aggregated), **not** from one surviving
group-lasso pruning. This is a **soft, collinearity-sensitive** signal (many pairs share SLC2A*/ABC*
genes) — corroborate across genes/cell types, don't over-trust. If direction becomes a primary
deliverable, we'd need to raise `group_reg`/`l1_reg` AND group finer (per-metabolite or per-pair
groups) — a departure from current defaults. Decision (this session): **keep one metabolite group #5
at default hyperparams; aggregate pairs→metabolite at read-time (not in-model)** — max info, can't
un-sum. Note also: harreman-directed orientation (D3's "revisit later") is **moot** — the `CT1→CT2`
arrow is a sorted-label artifact (05 §3), so harreman can't supply a real flux direction anyway.

Commit series (this session): `1602172` feat CU-1 metab group + loader → `68ec51d` feat CU-2 thread +
orphan relax → `c04ccad` feat beta-analysis + notebook.

## Session 2026-07-18 — neighborhood scores, tier reruns, metab_processing layout
- **Neighborhood scores** wired into `HarremanRunner` behind `compute_nbhd_scores=True`
  (`nbhd_scores.py`). Computed once in the cell-indep step, summarized per tier. See 05 §5a for
  the stats and the "not the ct statistic" warning. **Ran clean on Savio for melanoma, all 3 tiers.**
- **Tier reruns fixed**: `compute_gene_pairs` uses `uns.setdefault`, so tier 2+ reused tier 1's
  cell-type keys → `KeyError`. `run_cell_aware` pops `cell_type_pairs` / `gene_pairs_per_ct_pair`
  first. This is what makes looping tiers possible at all (05 §5a).
- **Corrected 05 §3 item 3**: both gene orders are built but filtered back out by
  `subset_gene_pairs`; only one order is actually scored. Conclusion (undirected) unchanged.
- **`metab_processing/` reorganized** into `Harreman/`, `Preprocess/`, `SpaceTravLR/`.
  `quick_start_metab.ipynb` deliberately **left in the old flat location and unmodified** (running
  on Savio, avoiding git conflicts) — its `metab_loader`/`beta_analysis` imports are stale until
  Foster moves it.
- **`run_full_harr.ipynb`** added — all datasets, all tiers, summaries, `.dataset_name` marker (05 §5b).
- Commit series: `fdef7d9` feat nbhd + setdefault fix → `8ef2d74` refactor layout →
  `4720919` feat run_full_harr.
- Local testing note: conda env **`spacetravlr_env`** has pandas/numpy/scanpy/anndata/torch, but
  **harreman is not installed locally** (Savio only) — anything importing harreman can only be
  checked statically.

## Session 2026-07-19 — beta_analysis rewritten (simple, tier-foldered)
The first beta-analysis module (`read_metab_beta_summary` → `aggregate_to_metabolite` →
`gene_set_score` + `gene_pair_cnp_weights`) was **thrown out** — too much machinery for what
Foster wants read out. **Those four functions no longer exist**; the pairs→metabolite rollup,
the `C_np` weighting and the signed gene-set ranking are all gone (do them downstream from the
CSVs if wanted). New `metab_processing/SpaceTravLR/beta_analysis.py`, ~150 lines:
- **A "tier" is an `adata.obs` cell-type column** (`Tier1/2/3`) and **a tier is a folder**:
  everything is written to `easy_download/metabtravlr_outputs/<tier>/`.
- `tier_means(betadata_dir, obs, tier, genes=, group=)` — the one shared primitive: group cells
  by `obs[tier]`, return mean/std/n per (gene, modulator, cell type). `group` filters to a
  modulator group; classification is by separator (`@` metab, `$` lr, `#` ltf, none → tf).
- `write_gene_pairs(...)` → `<tier>/gene_pairs.csv`, **metabolite (`@`) pairs only**, columns
  `gene, export, import, pair, cell_type, mean, std, n`. **Cell types stay as rows** (collapsing
  them would defeat the point of tiers) and **both orientations of a heterotypic pair stay as
  separate rows** — they're separate model coefficients (direction is read from magnitude, see
  the 2026-07-17 regularization finding).
- `write_histograms(..., bins=50, plot=False)` → `<tier>/histograms.csv` (`group, left, right,
  count`): distribution of those per-(gene, modulator, cell type) mean betas, one histogram per
  modulator group, pooled over genes and cell types. `plot=True` also writes `histograms.png`.
- `betas_to_adata(adata, betadata_dir, genes=, group='metab')` → `adata.obsm['beta_<gene>']`
  (cells x modulators, reindexed to `obs_names`, NaN for cells the gene wasn't fit on); column
  names in `adata.uns['beta_modulators'][gene]` since obsm drops labels.
- Notebook read-back cells updated to the new API. Tests rewritten (`tests/test_beta_analysis.py`,
  4 tests, known-answer). Verified on pandas **2.3.3** (`spacetravlr_env`) **and 3.0.0**, no
  FutureWarnings. Note real betadata parquets live on Savio — local `tmp/.../betadata` is empty
  and `Results/.../metabtravler_outputs/pair_summary.csv` is old-format output from the dead code.

## Session 2026-07-20 — harreman aggregate CCC memory fix (gene-pair chunking), CU-A–D
Fixed the OOM Foster hit **running `HarremanRunner`** at Xenium scale (decision **D9**; full
technical writeup in `05` **§5c**). Root cause: the two aggregate functions'
dense `(n_cells × n_gene_pairs)` matmul intermediates — *not* the per-cell `(cells, pairs, M)`
blowup of `05` §5 (that one was already handled via `nbhd_scores`), and **not** `M`-dependent.
Fix = gene-pair chunking, proven **bit-for-bit identical** to stock.

- **`metab_processing/Harreman/interacting_cell_scores_lowmem.py` → `cell_communication_lowmem.py`**
  (renamed; refs updated). Now holds **three** annotated-`# STOCK:`-diff drop-ins:
  `compute_interacting_cell_scores_lowmem` (CU-A, reformatted + replicated stock's float32 pval
  cast for exactness), `compute_cell_communication_lowmem` (CU-B), `compute_ct_cell_communication_lowmem`
  (CU-C). `harreman_funcs.py` calls all three (CU-D); `HarremanRunner(gene_pair_chunk_size=None)`
  auto-sizes (~50M-elem budget).
- **Two stock quirks reproduced for bit-identity** (report upstream): float32 pval cast (cell-indep);
  dtype-omitted `torch.zeros` → float32 `cs_gp` (ct). Both look like bugs.
- **Dev/review loop** (metab-dev + Opus metab-review per CU): every CU adversarially reviewed;
  CU-B stress-tested 600 configs, CU-C 120 — `cs` + entire non-parametric path (`pval`/`FDR`/`perm_cs`,
  the production-gating outputs) **exactly** bit-identical across all chunk sizes/seeds; parametric
  `Z` ≤ ~2 ULP float64 reduction noise (off the gating path). All local proof is **CPU**; a GPU/real-
  data gate `validate_lowmem_savio.py` must be run on Savio before production (reduction kernels
  could add ULP drift on CUDA — measure-zero exceedance-flip risk).
- **Tests** (all in `spacetravlr_env`, no real harreman — `tests/fixtures/fake_harreman/` vendors
  the real code byte-identically): `test_cell_communication_lowmem.py`, `test_cell_communication_agg_lowmem.py`,
  `test_cell_communication_ct_lowmem.py` (incl. a ct-specific-mask regression), `test_harreman_funcs_wiring.py`
  (AST guard against reverting to the OOM stock calls). Full suite 219 pass / 1 known-unrelated
  (`test_spawn_worker`, stale after the Savio SLURM refactor — left ignored per Foster).

## Session 2026-07-21 — per-cell nbhd GPU chunking (CU-E, Option B)
Implemented the last harreman GPU-memory bottleneck (decision **D10**; full writeup in `07` §10,
warning updated in `05` §5). The per-cell "neighborhood" `np` path OOM'd the Savio GPU at ≥600k
cells even after CU-A removed the `×M` axis, because it still held several dense `(n_cells × n_gp)`
float64 tensors at once.

- **CU-E = Option B, two-pass chunking** in `compute_interacting_cell_scores_lowmem`'s `np` branch
  (old block commented `# OLD:`, new `# NEW (CU-E ...)`, git-record style per Foster): Pass 1
  chunks the **gene-pair** axis (re-seed-per-chunk RNG replay, the proven CU-B/C trick), Pass 2
  chunks the **metabolite** axis (gather each chunk's union of gene pairs, remapped `sub_dict`,
  ~2× perm matmuls — sanctioned). GPU bounded to `(n_cells, chunk)`; full matrices assembled/stored
  on **CPU** (exact `uns` contract kept). BH run **once** over each full flattened p-value matrix.
- **Params:** `gene_pair_chunk_size` + `metabolite_chunk_size` (adaptive default `max(1,50M//n_cells)`)
  on the drop-in, threaded through `nbhd_scores.compute_nbhd_scores` **only** — `HarremanRunner`/
  `harreman_funcs.py` untouched (Foster's call on param surface).
- **Dev/review loop, Opus reviewer** (numeric+structural, per `07` §8). Review caught one **MAJOR**:
  a CPU/GPU **device seam** — observed `cs_m` was CPU-reduced while perm `cs_m` was GPU-reduced, so
  on CUDA a ≥3-pair metabolite's `.sum(dim=1)` could ULP-differ and flip an `x_m` exceedance
  (invisible to CPU tests). Fixed: observed `cs_m` now computed on `device` for both the stored
  value and the threshold; `use_p_shortcut` copies the parametric `cs_m` like stock. Two test/guard
  gaps also fixed (center=True sweep; `sparse.mm`-width memory guard + positive control).
- **Tests:** `tests/test_cell_communication_lowmem.py` sweeps `gp_chunk ∈ {1,2,n_gp,None}` ×
  `m_chunk ∈ {1,2,n_m,None}` vs **true stock**, incl. metabolite-spanning-chunks, shared-pair,
  ≥3-pair-metabolite, and centered paths. Full comm suite **60 pass / 636 subtests** (`spacetravlr_env`,
  fake harreman). All local proof is **CPU-exact**; GPU exactness is proven only on Savio via
  `validate_lowmem_savio.py` (section [3/3] now forces small nbhd chunks with
  `--nbhd-gp-chunk-size`/`--nbhd-m-chunk-size`).
- **Second device-parity bug — found by the real Savio run, fixed (2026-07-21).** First GPU run:
  aggregates non-parametric EXACT, per-cell `cs` EXACT, but per-cell `pval`/`FDR` off by exactly
  `2⁻²⁴` (0.5 float32-ULP). Fingerprint ⇒ integer exceedance counts identical; cause = the float32
  **division** `(x+1).float()/(M+1)` was done on **CPU** while stock does it on **GPU** (CUDA vs x86
  float32 divide disagree ≤1 ULP in `[0.5,1)`). Fixed: divide on `device` per chunk (elementwise ⇒
  chunk-invariant ⇒ still GPU-bounded), BH stays once on CPU. I first mis-read the fingerprint as a
  float64/float32 *dtype* mismatch (stale reference?) — checking **installed harreman v0.1.4** showed
  its per-cell pval also uses `.float()`, ruling that out. Lesson: "same device" covers float32
  **division**, not just reductions; verify a fingerprint against the *installed* package. See `07` §10.
- **CU-F — Pass 2 OOM at Xenium scale, fixed (2026-07-21).** A real `run_harr_all.py` run OOM'd in
  **Pass 2** (metabolite chunks): Pass 2 was sized by metabolite *count* but its footprint is
  `(n_cells, |gene-pair union|)`, and many-to-many pairs made a chunk's union ≈ full `n_gp` → the
  `(n_cells, n_gp)` tensor we chunk to avoid. Fix (bit-identity untouched — only chunk boundaries
  move): `_greedy_metabolite_chunks` bounds each chunk's gene-pair **union** to `element_budget //
  n_cells` (symmetric with Pass 1); inter-pass GPU cleanup + `empty_cache`; and an **automatic OOM
  fallback** (catch CUDA OOM → halve `element_budget` → retry ≤4×) so the batch self-tunes with NO
  per-dataset knobs (Foster's ask). Threaded `gene_pair_chunk_size`/`metabolite_chunk_size`/
  `element_budget` onto `HarremanRunner` (default None → automatic) as a manual escape hatch. Comm
  suite 70 pass; 21,888-check stress sweep vs stock, 0 mismatches. See `07` §11.
- **Status: CU-E + CU-F fixed in code, Savio re-validation pending** (expect per-cell `pval`/`FDR`
  EXACT and no Pass-2 OOM at ≥600k cells). **Not yet committed.**

## Local assets for dev/testing
- Demo data in `data/`: `Slidetags_human_tonsil.h5ad`, `Slidetags_human_melanoma.h5ad`,
  `SlideSeqV2_mouse_lymphnode.h5ad`, `XYZeqV2_mouse_kidney_replicate_{1,2}.h5ad`,
  `snrna_germinal_center.h5ad`; reference `cellchat_{human,mouse}.csv`, `{species}_base_grn.parquet`.
- Existing tests in `tests/` (unittest + synthetic `make_regression`/`np.random` adata, e.g.
  `test_spacetravlr.py`). Quick-start flow: `docs/source/quick_start.ipynb`.
- Metabolite example: `DataForClaude/documentation/easy_download/harreman_outputs/`.
