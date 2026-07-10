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

## Local assets for dev/testing
- Demo data in `data/`: `Slidetags_human_tonsil.h5ad`, `Slidetags_human_melanoma.h5ad`,
  `SlideSeqV2_mouse_lymphnode.h5ad`, `XYZeqV2_mouse_kidney_replicate_{1,2}.h5ad`,
  `snrna_germinal_center.h5ad`; reference `cellchat_{human,mouse}.csv`, `{species}_base_grn.parquet`.
- Existing tests in `tests/` (unittest + synthetic `make_regression`/`np.random` adata, e.g.
  `test_spacetravlr.py`). Quick-start flow: `docs/source/quick_start.ipynb`.
- Metabolite example: `DataForClaude/documentation/easy_download/harreman_outputs/`.
