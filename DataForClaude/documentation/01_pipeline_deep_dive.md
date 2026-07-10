# Pipeline deep-dive + surgical wiring map

> Exact call sites and line refs for the parts we touch, and the **precise, minimal edits**
> to add metabolites as a **new modulator group** (D6) with TF/LR kept optional (D7), train
> only genes of interest (D5), and **read the coefficients directly** (D1). Line numbers are
> from the current tree — re-confirm before editing.
>
> Read `00_overview.md` and `02_metab_integration_notes.md` first.

---

## 1. How a modulator becomes a betadata column (the path we extend)

Trace for one target gene, in `models/parallel_estimators.py`:

1. **Assemble modulators** — `SpatialCellularProgramsEstimator.__init__`:
   - `self.regulators` = TFs, from `grn.get_regulators(adata, target_gene)` (L484).
   - `init_ligands_and_receptors(...)` (L297) builds L–R pairs (CellChat, receptor-filtered)
     and L–TF pairs (NicheNet top-5). Returns `ligands/receptors/tfl_*`. `extra_lr` is
     appended here (L328-356) — this is the reference for our metabolite path.
   - `self.modulators = self.regulators + list(self.lr_pairs) + self.tfl_pairs` (L529),
     then `+ self.extra_modulators` (L546).
2. **Build the design matrix** — `init_data()` (L695):
   - received ligands (Gaussian-diffused) via `init_received_ligands` (L165).
   - `adata.uns['ligand_receptor']` = `received_ligand × receptor`, columns `lig$rec`
     (`ligands_receptors_interactions`, L614).
   - `adata.uns['ligand_regulator']` = `received_ligand × TF`, columns `lig#tf`
     (`ligand_regulators_interactions`, L634).
   - `self.train_df` = `[target]+regulators` joined with those two interaction frames
     (+ extra_modulators) (L757-765). Column order defines predictor order.
3. **Fit per cluster** — `fit()` (L864):
   - group-lasso group vector (L945):
     `groups = [1]*regulators + [2]*lr_pairs + [3]*tfl_pairs + [4]*extra_modulators`.
   - group-lasso anchors → CNN (`CellularNicheNetwork`) emits spatial β =
     `sigmoid(MLP)·anchors`.
4. **Emit betadata** — `get_betas()` (L828) / `betadata` property (L860):
   columns = `['beta0'] + ['beta_'+m for m in self.modulators]` (L856). Written to
   `betadata/{gene}_betadata.parquet` by `oracles.SpaceTravLR.run()` (L501).

**Reading back** — `beta.py::BetaFrame.__init__` (L114-122) classifies each `beta_<mod>`
column by separator: `$`→L–R, `#`→L–TF, else→TF. (Used by GeneFactory/perturbation, which
we don't run — but also the natural loader for reading.)

---

## 2. Where the external networks load (confirmed line refs)

| Network | Built at | Read at | File |
|---|---|---|---|
| CellOracle GRN (TF→target) | `spaceship.run_celloracle_` L244-294 (vendored `celloracle_tmp`, base GRN `SpaceTravLR_data/{species}_base_grn.parquet` via `load_base_GRN` L219) | `run_spacetravlr` L606 → `RegulatoryFactory(links=…)`; `tools/network.py::get_cluster_regulators` | `input_data/celloracle_links.pkl` |
| CellChat L–R DB | — (shipped) | `get_cellchat_db` (`tools/network.py` L17); `init_ligands_and_receptors` L322 | `SpaceTravLR_data/cellchat_{species}.csv` |
| NicheNet L–TF | `get_nichenet_links_` L495 (Zenodo dl) | `init_ligands_and_receptors` L380-397 | `input_data/tflinks.parquet` |
| COMMOT L–R filter (optional) | `run_commot_` L296 | masks pairs at setup | `input_data/communication.pkl`, `LRs.parquet` |

For metabolites we **replace COMMOT's role with harreman** (external, already computed) and
inject the significant transporter pairs as a new group.

---

## 3. Surgical change units (minimal, default-preserving)

Ordered; each is one dev/review loop unit. All new params default to *off* so existing
behavior is byte-identical when unused. **Prefer building the metabolite interaction frame in
our own script and passing it in, over expanding core signatures** — but the group/column
plumbing must live in the estimator.

### CU-1 — Add a metabolite modulator group to the estimator *(core, unavoidable)*
`models/parallel_estimators.py`:
- New optional ctor arg `metab_pairs: list[tuple[export, import]] | None = None` on
  `SpatialCellularProgramsEstimator` (default None → no change).
- Build metabolite interactions like L–R but with a **distinct separator** `@`:
  received_ligand(export) × import_expr → columns `export@import`. Reuse
  `received_ligands`/`ligands_receptors_interactions` (the export is diffused; ensure export
  genes are included in the received-ligand set). Both orientations already provided by the
  caller (D3), so no orientation logic here.
- Append `metab_pairs` to `self.modulators` **after** extra_modulators, and add
  `self.metab_pairs`.
- In `fit()` `groups`: append `[5]*len(self.metab_pairs)`. Keep the L–R/`$` and L–TF/`#`
  groups untouched.
- Betadata columns then naturally become `beta_<export>@<import>` (distinct, greppable).
- Guard: exclude pairs where export/import == target gene (mirror L494 logic); dedup.
- Tests: Tier-0 (column naming, both-orientation, homotypic, dedup) + Tier-1 (a `metab_pairs`
  arg yields a `beta_@` column and a group-5 entry) — see `03_...md`.

### CU-2 — Thread it through the trainer + orchestrator *(core, tiny)*
- `oracles.py::SpaceTravLR.__init__/run` (estimator built at L457): add optional
  `metab_pairs=None`, pass to the estimator. Default None.
- `spaceship.py::run_spacetravlr` (L567) + `SpaceShip`: optional `metab_pairs` passthrough.
  (We may bypass `SpaceShip` and drive `oracles.SpaceTravLR` directly from our script — TBD.)

### CU-3 — Train only genes of interest *(near-zero core edit; big tractability win, D5)*
- `oracles.py::SpaceTravLR` seeds `self.queue = OracleQueue(save_dir, all_genes=self.adata.var_names)` (L380).
  Option A (no core edit): from our script, override `st.queue = OracleQueue(save_dir, all_genes=target_genes)`
  before `st.run()`. Option B (1-line): optional `genes=None` ctor arg → `all_genes = genes or adata.var_names`.
- **Orphan gotcha (D7 interaction):** `run()` skips genes with `len(estimator.regulators)==0`
  (L475) — TFs only, even if metabolite/L–R modulators exist. For metabolite-only target
  genes we must relax this to `len(estimator.modulators)==0` (surgical, L475). Add a Tier-1
  test for the no-TF-but-metab case.

### CU-4 — Our-side scripts (no core edits) *(most of the code)*
In `metab_processing/` (or a new `metabtravlr/`):
- `load_harreman(tier)` → read `harreman_network.json` + `Tier{N}/[ct_ccc_results][cell_com_df_m].csv`;
  filter `selected==True`; expand metabolites → gene pairs; emit `metab_pairs`
  (both orientations, deduped, ∩ `adata.var_names`). (See `02_...md` §1 for schemas/bugs.)
- `load_gene_sets(path)` → `{label:[genes]}` (JSON/dict), validate vs `adata.var_names`.
- `read_metab_betas(betadata_dir, genes, metab_pairs)` → for each target gene G, load
  `G_betadata.parquet`, select `beta_<e>@<i>` columns → tidy `(gene, export, import, cell, β)`.
  (Prefer plain `pd.read_parquet` + column filter over `BetaFrame`, which would misclassify
  `@` columns — fine since we skip perturbation; note this in code.)
- `aggregate(scores, gene_sets)` → signed score
  `mean_{g∈positive} β̄(m,g) − mean_{g∈negative} β̄(m,g)`; β̄ over cells / cell types / region.
- SLURM driver mirroring `tutorial/launch.py`.

### CU-5 (later, only if scale demands) — O(N²) → sparse *(core, weightier)*
Replace dense `_gaussian_kernel_2d_batch` (L65) + `create_spatial_features` `cdist` (L244)
with `cKDTree` radius-neighbors (pattern already in `virtual_tissue.py`/`tools/analysis.py`).
Gate behind a flag; Tier-1 sparse-vs-dense equivalence test within tolerance. Needed for
100k–1M cells (see `02_...md` §3); not before.

---

## 4. What we deliberately do NOT touch
- `gene_factory.py`, `beta.py::splash`, `virtual_tissue.py` transitions/vector-field —
  perturbation/propagation, out of scope (D1). We only *read* betadata.
- The Sphinx `docs/` site.
- `BetaFrame`'s separator parser — unless we later want perturbation compatibility for
  metabolite columns (then extend L117-122 to recognize `@`).

## 5. Open confirmations before CU-1
- Separator choice `@` (vs `~`/`::`) — must avoid `$`, `#`, and characters unsafe in parquet
  column names / our downstream regexes.
- Whether metabolites reuse the L–R `radius`/Gaussian exactly, or take harreman-derived
  distances (start: reuse L–R secreted radius).
- Whether we drive training via `SpaceShip` or directly via `oracles.SpaceTravLR` from our
  script (affects CU-2 surface).
