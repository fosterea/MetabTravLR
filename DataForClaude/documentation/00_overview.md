# SpaceTravLR — Developer Overview

> Purpose of this doc: give us (Claude + Foster) a shared, accurate **mental model of
> the code** so we can adapt it. This is the "map." A companion doc
> (`01_pipeline_deep_dive.md`) drills into exact call sites (e.g. where CellOracle loads
> and when it is referenced).
>
> Source of truth: the preprint (`../paper_fulltext.txt`, extracted from
> `../Space_TravLR_Preprint.pdf`) + the code under `src/SpaceTravLR/`.
> Package version `0.1.16` (setup.py). Note: the repo was renamed from **SpaceOracle**
> → **SpaceTravLR**; both names appear in code, paths, and CI.

> ### 🎯 Our project's scope (read this first)
> We are adapting SpaceTravLR to ask **which metabolites** (each defined by an
> import/export **gene pair**) **modulate which sets of target genes**, spatially.
> Two decisions from Foster shape how we read this codebase:
> 1. **Metabolites map onto the ligand–receptor abstraction.** An *export* gene behaves
>    like a diffusible ligand (source); an *import* gene behaves like a receptor (sink).
>    So the machinery that builds and learns coefficients for L–R edges is the machinery
>    we extend. (§7)
> 2. **We will analyze the learned coefficients (β's) directly — NOT run perturbation
>    prediction.** The trained model already emits, per cell and per target gene, a
>    spatially-resolved coefficient for every modulator (TF, L–R, L–TF). For a metabolite
>    edge, **that coefficient _is_ the answer** ("how much does this metabolite modulate
>    this target gene, here?"). This means **Phase 1 (training → `betadata` coefficients)
>    is our whole focus; Phase 2 (`GeneFactory.perturb()`, the propagation engine) is out
>    of scope** and we can largely ignore it beyond understanding what betadata contains.
> Wherever this doc discusses perturbation, treat it as *context for how the coefficients
> are meant to be used*, not something we will run.

---

## 1. What SpaceTravLR does (one paragraph)

SpaceTravLR takes **spatial transcriptomics** (ST) data — a gene-expression matrix plus
an (x, y) location and a cell-type label for every cell/spot — and learns an
**interpretable, spatially-varying linear model** of each gene's expression as a function
of its upstream regulators. Regulators come in three biological flavors: **transcription
factors (TFs)** (cell-intrinsic), **ligand–receptor (L–R) pairs** and **ligand–TF (L–TF)
pairs** (cell-extrinsic / signaling, distance-weighted from neighboring cells). Once
trained, the model can **simulate in-silico perturbations** (knock out / over-express any
TF, ligand, or receptor in any chosen set of cells) and **propagate** the effect through
the gene network and across space (cell → neighbor via ligands), predicting a whole-
transcriptome shift per cell and the resulting cell-state transitions. The novel output is
**functional microniches**: spatial regions where the *same* cell type has *different*
perturbation outcomes because of its neighborhood.

**Why this matters for our project:** the L–R / L–TF machinery is a generic
"gene-A-in-a-neighbor affects gene-B-in-this-cell, weighted by distance" engine. Our
**metabolite import/export gene pairs** map onto exactly this abstraction (export gene =
"ligand"/source, import gene = "receptor"/sink), and the "sets of genes affected" are the
downstream targets the model already predicts shifts for. See §7.

---

## 2. The two-phase mental model

Everything in the codebase is one of two phases. Keep them separate in your head.

```
                 ┌─────────────────────────── PHASE 1: TRAIN ───────────────────────────┐
 raw AnnData ──► setup_ (preprocess + build networks) ──► per-gene model fit ──► betadata
                 (CellOracle, COMMOT, NicheNet, CellChat)   (CNN → spatial β's)   *.parquet
                                                                                     │
                 ┌─────────────────────────── PHASE 2: SIMULATE ────────────────────┘────┐
 betadata ──► load β's ──► perturb(target, level, cells) ──► 4-step chain-rule ──► simulated
              (GeneFactory)                                    propagation           GEX + transitions
```

- **Phase 1 (train), one model per gene.** For each of ~5000 highly-variable genes,
  SpaceTravLR fits a *separate* spatially-weighted regression. The learned output is not a
  single coefficient but a **CNN that emits per-cell, per-modulator coefficients ("betas")
  as a function of the cell's spatial neighborhood.** These are saved to
  `betadata/{gene}_betadata.parquet` (a `cells × modulators` table of β values).
  Training is embarrassingly parallel across genes and is built to run as many SLURM
  workers pulling from a shared job queue (see `OracleQueue`, §4).

- **Phase 2 (simulate), no training.** `GeneFactory` loads all the saved betadata and runs
  `perturb()`. This does **not** retrain anything — it reads the β's, applies the
  perturbation via analytic derivatives (chain rule), and iterates 4 propagation steps to
  spread the effect through the network and across space.

> Mental shortcut: **Phase 1 produces β(location) for every gene. Phase 2 does linear
> algebra with those β's to answer "what if I change gene X in these cells?"**

---

## 3. The core math, in plain terms

The paper's model (Methods, `paper_fulltext.txt` ~p17–23). You don't need the LaTeX to work
in the code; you need this intuition:

**Target gene expression = intercept + Σ(β_TF · TF) + Σ(β_LR · received_ligand · receptor) + Σ(β_LTF · received_ligand · TF)**

- Each **β is a function of the cell's (x, y)** — that's the whole point. Two
  transcriptionally-identical cells in different neighborhoods get different β's, hence
  different perturbation outcomes. The β's are produced by a small **CNN** that looks at a
  64×64 "spatial proximity grid" (SPG) of the cell's neighborhood (per cell type), followed
  by an MLP whose output width = number of modulators + 1 intercept.
- **"received_ligand"** = ligand expression of neighbors within a radius, **Gaussian-
  distance-weighted** (secreted signaling radius ≈ 200–300 µm; contact signaling ≈ 30–50 µm).
  This is the spatial coupling. Receptors are used raw (surface-bound, not diffusible).
- The β's are **split** into a non-spatial part (learned by a **sparse group-lasso**, which
  zeroes out whole groups of uninformative predictors) times a spatial part (the CNN):
  `β(u,v,c) = β_nonspatial(c) * β_spatial(u,v,c)`. Group-lasso groups = {TFs, L–R, L–TF,
  extra}. This is why coefficients are interpretable and sparse.

**Perturbation = derivatives.** Because the model is linear in the modulators, the effect of
perturbing a gene is just the coefficient (chain rule for multi-hop). Perturbing a **ligand**
has effect ∝ its receptor's expression; perturbing a **receptor** ∝ received ligand; etc.
(`paper_fulltext.txt` p20–21). Multi-hop effects (gene → gene → gene, up to **N=4** hops)
compose by multiplying coefficient tensors. This is what `perturb()` implements.

**Readout.** From the simulated whole-transcriptome shift per cell, SpaceTravLR computes a
**cell→cell transition probability matrix** (softmax over correlation of the shift with the
observed cell-cell expression differences — same idea as CellOracle), projects it into UMAP
as a **vector field**, and scores functional impact by the **inner product** of the
perturbation vector with a reference gradient (pseudotime, gene-set score, etc.).

---

## 4. Code architecture map

All paths under `src/SpaceTravLR/`. Whimsical names (`spaceship`, `oracles`, `astronomer`,
`visionary`, `virtual_tissue`) — the table says what each really is.

| File | Real role | Key classes / entry points |
|---|---|---|
| `spaceship.py` | **User-facing orchestrator** (README API). Wraps the whole Phase-1/Phase-2 flow. | `SpaceShip.setup_()`, `.run_spacetravlr()`/`.fit()`, `.spawn_worker()` (SLURM), `.setup_perturbations()`, `.perturb()`, `.is_everything_ok()` |
| `oracles.py` | **Phase-1 training driver** + preprocessing base class + the parallel job queue. | `BaseTravLR` (PCA/MAGIC imputation), `SpaceTravLR` (the training loop `.run()`), `OracleQueue` (file-lock job queue for parallel workers) |
| `models/parallel_estimators.py` | **The heart of Phase 1.** Assembles each gene's modulators (TF + L–R + L–TF + extra) and fits the per-cluster CNN + group-lasso. | `SpatialCellularProgramsEstimator`, `init_ligands_and_receptors()`, `received_ligands()`, `init_received_ligands()` |
| `models/pixel_attention.py` | The actual neural nets that emit spatial β's. | `CellularNicheNetwork` (CNN, default), `CellularViT` (Vision Transformer alt) |
| `models/spatial_map.py` | Builds the 64×64 spatial proximity grids per cell. | `xyc2spatial`, `xyc2spatial_fast` |
| `models/*_estimators.py`, `bayesian_linear.py`, `vit_blocks.py` | Supporting model variants / building blocks. | (see agent map) |
| `tools/network.py` | **Loads the GRNs.** Wraps the pickled CellOracle `Links` and the CellChat DB. | `CellOracleLinks` (base), `RegulatoryFactory`, `DayThreeRegulatoryNetwork` & friends (one per dataset), `get_cellchat_db()`, `expand_paired_interactions()` |
| `gene_factory.py` | **The heart of Phase 2.** Loads betadata, runs perturbation + propagation. | `GeneFactory.perturb()`, `.perturb_batch()`, `.genome_screen()`, `.load_betas()` |
| `beta.py` / `beta_utils.py` | Betadata I/O + the **"splash"** step that converts `cells×modulators` β's into `cells×genes` gene-gene derivatives. | `BetaFrame`, `Betabase` (see agent map for details) |
| `virtual_tissue.py` | Transition probabilities + vector-field / functional-alignment readout (the "microniche" layer). | (see agent map) |
| `astronomer.py`, `visionary.py` | Higher-level analysis / orchestration helpers. | (see agent map) |
| `tools/` | Utilities: `knn_smooth.py`, `analysis.py`, `data.py`, `utils.py`, `iwanthue.py`. | `is_mouse_data`, `gaussian_kernel_2d`, `scale_adata`, … |
| `plotting/` | All figures (cartography, niche, sankey, beta_maps, shift, quantify…). | — |
| `callbacks/` | Simulation callbacks / fixtures (incl. a simulator for validation on synthetic data). | — |
| `SpaceTravLR_data/` (sibling pkg) | **Shipped reference data**: `{human,mouse}_base_grn.parquet` (CellOracle base GRN), `cellchat_{human,mouse}.csv` (L–R DB). | — |
| `celloracle_tmp/` (sibling pkg) | **Vendored, trimmed CellOracle** used only at setup to build the GRN links. | `co.Oracle()` |

### The training loop in one glance (`oracles.py::SpaceTravLR.run`)
```
while queue has genes:
    gene = queue.next()                       # random/priority pick, file-locked
    est = SpatialCellularProgramsEstimator(gene, grn=..., tflinks=..., radius=...)
    if est.regulators == []:  queue.mark_orphan(gene); continue   # no known upstream → skip
    est.fit(...)                              # per-cluster: group-lasso init → CNN training
    est.betadata.to_parquet(betadata/{gene}_betadata.parquet)
```
`OracleQueue` coordinates many processes purely through files in the `betadata/` dir:
`*.lock` = in progress, `*.parquet` = done, `*.orphan` = untrainable. That is how
`spawn_worker()` scales across SLURM nodes with no central server.

---

## 5. End-to-end data flow (with the artifacts on disk)

```
INPUT: adata (AnnData): .X counts, .obsm['spatial'] (x,y), .obs['cell_type']

SpaceShip.setup_(adata):
  process_adata_        → normalize/log1p, MAGIC impute (per cell type) → 'imputed_count' layer
                          save → output/input_data/_adata.h5ad
  run_celloracle_       → celloracle_tmp.Oracle + base_grn.parquet → per-cluster Bayesian-Ridge GRN
                          → output/input_data/celloracle_links.pkl   (dict{cluster: DF[source,target,coef_mean,p]})
  run_commot_ (optional)→ COMMOT optimal-transport CCC test → filters L–R pairs (p<0.3)
                          → output/input_data/communication.pkl, LRs.parquet
  get_nichenet_links_   → download ligand_target_{species}.parquet (NicheNet L–TF matrix)
                          → output/input_data/tflinks.parquet

SpaceShip.run_spacetravlr()  (a.k.a. .fit(); or launch.py under SLURM via spawn_worker):
  for each gene → SpatialCellularProgramsEstimator.fit()
       modulators = TFs (from celloracle_links) + L–R (CellChat, receptor-thresh + COMMOT mask)
                    + L–TF (NicheNet top-5 per TF) + optional extra
       → betadata/{gene}_betadata.parquet     (cells × modulator β's)
  + betadata/run_params.json  (all hyperparams; used to re-instantiate GeneFactory)

SpaceShip.setup_perturbations(adata) → GeneFactory.from_json(run_params.json) + load_betas()
SpaceShip.perturb(target, gene_expr, cells):
  GeneFactory.perturb → 4× { splash β's → gene-gene coeffs; recompute received ligands;
                             Δ_simulated = (Δligand + Δtarget) · Θ; clip to observed range }
  → simulated cells × genes expression (optionally saved as parquet / adata layer)
```

Output tree (from README):
```
output/
├── input_data/  _adata.h5ad, celloracle_links.pkl, communication.pkl, LRs.parquet, tflinks.parquet
├── betadata/    {GENE}_betadata.parquet ×N, run_params.json
└── logs/        training_TIMESTAMP.log
```

---

## 6. Where the external tools enter (short version — deep-dive doc will expand)

| Tool | What it provides | Where it's used | When |
|---|---|---|---|
| **CellOracle** (vendored as `celloracle_tmp`) | Base GRN → per-cluster TF→target links with coefficients & p-values | `spaceship.run_celloracle_()` builds them; `tools/network.py` classes *read* the pickle | Built **once at setup**; only the pickle is read during train/perturb. `import celloracle` is commented out everywhere except `run_celloracle_`. |
| **CellChat** | Curated ligand–receptor pairs (`SpaceTravLR_data/cellchat_{species}.csv`) | `get_cellchat_db()`, `init_ligands_and_receptors()` | Every gene fit (defines candidate L–R modulators) |
| **NicheNet** | Ligand→target regulatory potential matrix (`ligand_target_{species}.parquet`) | `get_nichenet_links_()`, `init_ligands_and_receptors()` | Defines L–TF modulators (top-5 ligands per TF) |
| **COMMOT** | Optimal-transport significance test for L–R pairs | `spaceship.run_commot_()` (optional) | Setup only; masks non-significant L–R pairs |
| **MAGIC** (`magic-impute`) | Expression imputation/smoothing on the manifold | `BaseTravLR.impute_clusterwise()` | Preprocessing |
| **group-lasso**, **torch/pyro** | Sparse group regression; CNN/ViT | `parallel_estimators.fit()` | Training |

> **Key takeaway for us:** at runtime the "GRN" is just pickled DataFrames of
> `source, target, coef_mean, p`. CellOracle the package is only a setup-time dependency.
> That makes it feasible to swap/augment the network (e.g. add metabolite edges) without
> touching CellOracle itself.

---

## 7. Our adaptation: metabolite edges as modulators, read out via coefficients

This is the reason we're reading the code. Because we're **analyzing coefficients directly
(not perturbing)**, the plan has two parts: (A) get metabolite import/export pairs *into the
model as modulators* so the trainer learns a β for them, and (B) *read those β's* back out
of the betadata against our target gene sets.

### A. Getting a metabolite edge into training (so a β is learned for it)

A metabolite = (export gene = ligand/source, import gene = receptor/sink). Two candidate
hooks, both in `models/parallel_estimators.py`:

1. **`extra_lr` parameter** (already exists) — `init_ligands_and_receptors()` and
   `init_received_ligands()` accept `extra_lr: list[tuple[ligand, receptor]]` and append
   these to the CellChat L–R table as "Secreted Signaling" (Gaussian distance-weighted).
   Each pair becomes an `export$import` modulator column and gets its own learned β per
   cell per target gene. This is the most faithful representation of a diffusible
   metabolite and reuses the entire spatial-weighting path.
   ⚠️ **Caveat:** `extra_lr` is plumbed into the *estimator* but the training driver
   (`oracles.py::SpaceTravLR.run`, line 457) and `SpaceShip` do **not** currently pass it
   through. **Wiring `extra_lr` end-to-end (SpaceShip → SpaceTravLR → estimator) is likely
   our first code change.**

2. **`extra_modulators`** — adds arbitrary genes as a 4th (non-spatial) predictor group.
   Use if a metabolite should act cell-intrinsically instead of via diffusion. Less likely
   what we want, but cheap to keep in mind.

Design question to resolve with Foster: is the metabolite's "received ligand" the **export
gene's expression** (proxy for local production) diffused over the radius, with the β then
multiplied by the **import gene's expression** (the current `L$R` = received_ligand ×
receptor formulation)? That's what `ligands_receptors_interactions()` computes today
(`parallel_estimators.py:614`). We likely reuse it verbatim.

### B. Reading the coefficients back out (the actual analysis)

The trained output per gene is `betadata/{GENE}_betadata.parquet`: a **cells × modulators**
table. Column naming convention (`beta.py::BetaFrame`, confirmed):
- `beta0` — intercept
- `beta_<TF>` — a transcription-factor coefficient
- `beta_<ligand>$<receptor>` — an **L–R coefficient** ← *our metabolite edges live here*
- `beta_<ligand>#<TF>` — an L–TF coefficient

So to answer **"which metabolite modulates gene set S, where?"**:
- Train with our metabolite pairs added (part A). Every target gene G ∈ S now has a
  `beta_<export>$<import>` column in `G_betadata.parquet` giving the spatially-resolved
  coefficient of that metabolite on G, per cell.
- Aggregate those β's across G ∈ S (and/or across cells / cell types / spatial regions) to
  rank metabolites by their effect on the gene set. This is a **read + groupby over parquet
  files**, not a simulation.

> Written before D6: metabolites ended up as their **own** modulator group with an `@`
> separator (`beta_<export>@<import>`), not reusing `$`. The read-out is implemented in
> `metab_processing/SpaceTravLR/beta_analysis.py` — see 04's 2026-07-19 entry.
>
> **⚠️ Updated by D11 (2026-08-11):** the column is now **one summed column per metabolite**,
> `beta_metab@<name>` (sum over the metabolite's transporter pairs, both orientations), not one
> per gene pair. `read_metab_betas`/aggregation below and the `beta_<export>@<import>` phrasing
> throughout §7 are superseded — see `04_decisions_and_state.md` D11.

Relevant existing code to reuse for the read-out (no perturbation needed):
- `beta.py::BetaFrame.from_path()` / `Betabase.load_betas_from_disk()` — load betadata.
- `beta.py::Betabase.collect_interactions()` — aggregates β's per cell type (agent-confirmed).
- `plotting/niche.py::get_modulator_betas`, `plotting/beta_maps.py::plot_spatial` — pull and
  spatially visualize a single modulator's β across cells (this is basically our figure).

> **Bottom line:** our pipeline is `add metabolite edges → run_spacetravlr (train) → read
> beta_<export>$<import> columns from the betadata parquets over our gene sets`. We touch
> the estimator's modulator assembly and the betadata read/aggregate layer; we do **not**
> touch `GeneFactory`/`perturb`.

Open inputs Foster will provide (define file formats for these next):
- metabolite ↔ (export gene, import gene) mapping.
- the target gene sets S to score against.

---

## 8. Glossary / naming gotchas

- **SpaceOracle vs SpaceTravLR** — same project, renamed. Old name persists in CI badge,
  some data paths (`/ix/djishnu/alw399/SpaceOracle/...` — absolute HPC paths, will not
  exist locally), and hard-coded dataset classes in `tools/network.py`.
- **"betadata"** — the per-gene `cells × modulators` table of learned coefficients (β).
- **"splash"** (`beta.py`) — converting modulator-space β's into gene-space gene-gene
  derivatives by distributing ligand terms back onto their component genes (chain rule).
- **"modulators"** — the union of a target gene's predictors: TFs ∪ L–R pairs ∪ L–TF pairs
  ∪ extras. Column names encode type: `ligand$receptor` (L–R), `ligand#TF` (L–TF),
  bare gene (TF/extra).
- **"regulators"** — specifically the TF subset (from the CellOracle links).
- **`OracleQueue` / orphan / lock** — the file-based parallel job scheduler; an "orphan" is
  a gene with no trainable upstream network.
- **Dataset-specific GRN classes** (`DayThreeRegulatoryNetwork`, `MouseKidneyRegulatoryNetwork`,
  `HumanTonsilRegulatoryNetwork`, …) — each just points at a different `celloracle_links*.pkl`
  and cluster-label map. For a new dataset, prefer the generic `RegulatoryFactory(links=...)`
  (that's what `SpaceShip.run_spacetravlr` uses).

---

## 9. What the deep-dive doc (`01_...`) will cover next

Reordered around our coefficient-analysis scope:
- **The betadata contract** (highest priority): exact `{gene}_betadata.parquet` schema, how
  `pixel_attention.CellularNicheNetwork` produces β = `sigmoid(MLP(spatial)) · anchors`
  (so β's are bounded and anchored to the group-lasso solution — important for interpreting
  magnitudes), and how `BetaFrame`/`Betabase` load and index them.
- **Exact call graph & line refs for CellOracle**: `run_celloracle_` internals, base-GRN
  format, `Links` schema, how `get_cluster_regulators` builds per-cluster TF masks.
- **How a modulator column comes to exist**: the full path from `init_ligands_and_receptors`
  → group-lasso groups → CNN output columns → betadata columns, so we know exactly where a
  metabolite `export$import` column is created and named.
- **The concrete metabolite plan**: the exact functions to modify to thread `extra_lr`
  end-to-end, plus the read/aggregate recipe over gene sets S, plus input file formats.
- (Lower priority / context only) `beta.py::splash()` and `GeneFactory.perturb()` — documented
  briefly so we understand what betadata is *nominally* for, but not on our critical path.
```
