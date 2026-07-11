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
  (Audit whether MAGIC imputation + `xyc2spatial` add further O(N²) before the CU-5 work.)

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
- **CU-1 (metabolite group) / CU-5 (O(N²) sparse):** not started. CU-5 is the real large-data
  enabler.

Commit series (this session): `2e523aa` feat gene-focus → `2699947` fix activation → docs.

## Local assets for dev/testing
- Demo data in `data/`: `Slidetags_human_tonsil.h5ad`, `Slidetags_human_melanoma.h5ad`,
  `SlideSeqV2_mouse_lymphnode.h5ad`, `XYZeqV2_mouse_kidney_replicate_{1,2}.h5ad`,
  `snrna_germinal_center.h5ad`; reference `cellchat_{human,mouse}.csv`, `{species}_base_grn.parquet`.
- Existing tests in `tests/` (unittest + synthetic `make_regression`/`np.random` adata, e.g.
  `test_spacetravlr.py`). Quick-start flow: `docs/source/quick_start.ipynb`.
- Metabolite example: `DataForClaude/documentation/easy_download/harreman_outputs/`.
