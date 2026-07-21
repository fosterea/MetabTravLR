# 05 — harreman reference (what it does, from the source)

Durable notes on **harreman**, the package Foster runs to score metabolite crosstalk between
cells. Written by reading harreman's `tools` source + the `HarremanRunner` wrapper
(`metab_processing/Harreman/harreman_funcs.py`) and verifying against the `easy_download` example.
**Read this before touching anything that consumes harreman output.** It is the source of
truth for the arrow semantics, the output tables, and the known GPU-OOM in the per-cell
("neighborhood") analysis.

---

## 1. What harreman is

harreman infers **spatial cell–cell communication (CCC)** from spatial transcriptomics
(here Xenium). It has two databases: **`LR`** (ligand–receptor) and **`transporter`**
(metabolite importers/exporters). We use **`transporter` only** — the question is which
**metabolites** are exchanged between neighboring cells, read out through **transporter gene
pairs** (SLC*/ABC*/AQP*/ATP* families).

Core idea: a metabolite is "communicated" between two spatially-adjacent cells if one cell
expresses a transporter for it and its neighbor expresses a (possibly the same) transporter
for it. harreman scores this as a spatially-weighted co-expression of the metabolite's
transporter gene pairs, and tests significance by permutation (non-parametric) and/or an
analytic Z-score (parametric).

The method is a spatial-CCC cousin of **Hotspot** (harreman vendors `..hotspot` models);
the statistical models `danb` / `bernoulli` / `normal` / `none` used to standardize counts
come from there.

---

## 2. The pipeline (functions in call order)

Wrapper = `HarremanRunner` in `metab_processing/Harreman/harreman_funcs.py`. It calls these harreman
`tl`/`pp` functions:

| Step | Function | What it does |
|---|---|---|
| DB | `pp.extract_interaction_db(adata, database='transporter', extracellular_only=True)` | Loads the transporter interaction DB into `adata.varm['database']` (genes × metabolites, values ∈ {−1 IMP, +1 EXP, 2 IMP-EXP}) and `adata.uns['database']='transporter'`. |
| Filter | `tl.apply_gene_filtering(model='danb', ...)` | Optional gene filters: feature-elimination (sparsity), spatial-autocorrelation, per-cell-type expression, per-cell-type DE. Foster runs with `autocorrelation_filt=False` (i.e. keep all DB genes). Writes `uns['filtered_genes']`, `uns['filtered_genes_ct']`. |
| Graph | `tl.compute_knn_graph(compute_neighbors_on_key='spatial', n_neighbors=5, weighted_graph=False)` | Spatial KNN → `adata.obsp['weights']`. `weights[i,j] > 0` means **j is a spatial neighbor of i** (row = focal cell i, col = neighbor j). Not symmetric in general. |
| Pairs | `tl.compute_gene_pairs(ct_specific=?, cell_type_key=?)` | Builds the metabolite→gene-pair map and, when `ct_specific`, the cell-type-pair enumeration. **See §3 — this is where the "arrow" is decided.** |
| CCC (agnostic) | `tl.compute_cell_communication(model='danb', M=1000, test='both', layer_key_p_test='counts', layer_key_np_test='log_norm')` | Cell-type-**independent** scores + significance → `uns['ccc_results']`. |
| CCC (aware) | `tl.compute_ct_cell_communication(..., cell_type_key, subset_gene_pairs=<sig from agnostic>, fix_gp=False)` | Cell-type-**aware** scores stratified by cell-type pair → `uns['ct_ccc_results']`. |
| Select | `tl.select_significant_interactions(test='non-parametric', ct_aware=?, threshold=0.05)` | Adds `selected` bool + `_sig` subset tables. **NP significance = `FDR_np < thr AND C_np > 0`.** |

`HarremanRunner.run_harreman(cell_type_col)` runs agnostic → aware(for that annotation column
= "tier") → saves CSVs. Each **Tier** is just one `cell_type_col` (a different `adata.obs`
annotation granularity); harreman has no notion of "tiers" — that is Foster's naming.

Layers: parametric test uses raw **`counts`**; non-parametric uses **`log_norm`**
(normalize_total 1e4 + log1p). We care about **non-parametric FDR (`FDR_np`)**.

---

## 3. THE ARROW: `Cell Type 1 → Cell Type 2` is NOT a direction

**This is the single most important gotcha.** For a cell-type pair `(CT1, CT2)` and gene
pair `(Gene 1, Gene 2)`, the cell-type-aware score is (from
`run_ct_cell_communication_analysis` + `create_weights_ct_pairs`):

```
score(CT1, CT2) = Σ_{i ∈ CT1, j ∈ CT2}  Gene1_expr(i) · W(i, j) · Gene2_expr(j)
```

- `W(i,j)` = spatial weight, **CT1 = row/focal cell i, CT2 = col/neighbor cell j**. `Gene 1`
  is read in CT1, `Gene 2` in CT2. (`compute_gene_pairs` keeps a `(var1,var2)`×`(ct1,ct2)`
  combo iff `var1 ∈ filtered_genes_ct[ct1]` and `var2 ∈ filtered_genes_ct[ct2]`.)
- Uses **only `W`** (not `W + Wᵀ`) — contrast the cell-type-**independent** path, which
  **symmetrizes** (`(counts_1·WX2) + (counts_1·WtX2)`) and is explicitly undirected.

Despite that anchoring, the ordering **cannot be read as flux direction**, for three
independently-sufficient reasons:

1. **Only the sorted ordering is ever computed.** `compute_gene_pairs` builds ct pairs with
   `itertools.combinations_with_replacement(sorted_cell_types, 2)` when `fix_ct` is unset
   (Foster's runs). Each *unordered* pair appears **once**, in sorted-label order; the reverse
   is never tested. **Verified**: the `(CT1,CT2)` set in every tier CSV == `combinations_with_
   replacement` of the sorted labels. Because `"other"` sorts last (lowercase), `(other → T)`
   simply never exists — it was *not tested*, not *found insignificant*.
2. **Genes carry no direction.** Every transporter gene is typed **`IMP-EXP`** (bidirectional)
   — verified 2042/2042 type slots. No pure IMP/EXP anywhere.
3. **Both gene orders are built, then filtered back out** (corrected 2026-07-18; this item
   previously claimed both orders contribute). `compute_gene_pairs` does union
   `combinations_with_replacement` **and** `permutations` for transporters when `ct_specific`,
   so `gene_pairs_per_ct_pair` holds both `(a,b)` and `(b,a)`. But `run_ct_cell_communication_
   analysis` then keeps only pairs present in `gene_pairs` — and `HarremanRunner` passes
   `subset_gene_pairs=<sig pairs from the cell-indep run>`, which is single-order. The reverse
   pairs are dropped by `if pair not in gene_pairs: continue`. Net effect on our runs: **only
   one gene order is scored.** The conclusion (undirected) is unchanged; the mechanism is not.

**Correct reading:** `score(CT1,CT2)` is an **undirected** measure of how strongly a
metabolite's transporters are **spatially co-expressed across the CT1–CT2 neighbor interface**
(or *within* a type, on the diagonal `CT==CT`). You may say *"metabolite M is exchanged at the
Effector-CD8 ↔ other interface"*; you may **not** say *"M flows into/out of T cells."*

To get any directionality you'd have to rerun with **`fix_ct='all'`** (→ `itertools.product`,
both orderings) and compare `score(X,Y)` vs `score(Y,X)` — but with `IMP-EXP` genes even that
asymmetry is spatial-anchoring, not metabolite flux. Not recommended as "direction".

`metab_processing/Harreman/harreman_summary.py` encodes all of the above: undirected `A–B` interfaces
and `A (self)` diagonals; `tcell_involvement ∈ {within_Tcell, within_other, Tcell_interface,
non_Tcell_interface, cell_type_independent}`.

---

## 4. Output data model

### 4a. `harreman_network.json` (from `save_harreman_network`, DB baseline)
- `num_transporter_genes` (100), `num_metab` (160), `num_gp` (416).
- `metabolite_pair_counts`: `{metabolite: n_gene_pairs}` (1–91).
- `gp`: list of `[geneA, geneB]` pairs. `gp_per_metabolite`: `{metabolite: {gene_pair:[[g1,g2],…],
  gene_type:[["IMP-EXP","IMP-EXP"],…]}}`.
- **Gene pair ↔ metabolite is many-to-many** (139/416 pairs serve >1 metabolite; e.g.
  `SLC2A3–SLC2A4` → glucose, DHA, dehydroascorbate). So a significant *gene pair* does not
  pin a single metabolite. Typo key: `transporter_gense`.

### 4b. Result tables (per `select_significant_interactions`)
Two grains × two tests. Grain: **`gp`** (gene-pair) and **`m`** (metabolite, = sum of its gene
pairs' scores via `compute_metabolite_cs`). Test columns: parametric `C_p, Z, Z_pval, Z_FDR`;
non-parametric `C_np, pval_np, FDR_np`. `selected` bool + `_sig` filtered copy added by
`select_significant_interactions`.

| Table (in `uns`) | Written CSV (by `save_harreman_outputs`) | Rows |
|---|---|---|
| `ccc_results['cell_com_df_m']` | `[ccc_results][cell_com_df_m].csv` | metabolite (no cell types) |
| `ccc_results['cell_com_df_gp_sig']` | `[ccc_results][cell_com_df_gp_sig].csv` | gene pair (no cell types) |
| `ct_ccc_results['cell_com_df_m']` | `Tier*/[ct_ccc_results][cell_com_df_m].csv` | (CT1, CT2, metabolite) |
| `ct_ccc_results['cell_com_df_gp_sig']` | `Tier*/[ct_ccc_results][cell_com_df_gp_sig].csv` | (CT1, CT2, Gene1, Gene2) |

The ct `_gp_sig` file **is** a real gene-pair table (an earlier version of the wrapper
duplicated `_m` there; current code fixed it — verified). `_m_sig` = `selected==True` subset.
NP `selected` requires `FDR_np < thr AND C_np > 0` (positive communication only).

### 4c. Our summary (`harreman_summary.py`)
`summarize_harreman_folder(path, background_label='other', sample_id=None) → (master, genepairs)`.
- **master** (1 row/metabolite): `n_gene_pairs`, `transporter_genes`, `global_significant`,
  `global_FDR_np`, `n_sig_gene_pairs_global`, and per tier `_n_sig_pairs`, `_interactions`
  (all undirected, compact), `_within_Tcell`, `_Tcell_interfaces`, `_tcell_involved`.
- **genepairs** (1 row/significant `tier × ct-pair × gene pair`): `tcell_involvement`,
  `metabolites` (many-to-many join), `FDR_np, pval_np, C_np`. `tier='global'` rows = the
  cell-type-independent gp table.
- Built for one folder; a later multi-sample manager concats with `sample_id`.

---

## 5. Per-cell "neighborhood" analysis + the GPU-OOM ⚠️

Beyond the aggregate CCC scores, harreman can compute **per-cell interacting-cell scores** —
for **each individual cell**, its contribution to each gene-pair / metabolite interaction
(i.e. *which cells* are doing the talking, not just whether the interaction exists). These are
**not** run by the current `HarremanRunner`; they are separate `tl` functions:

- `compute_interacting_cell_scores(...)` — cell-type-**independent**, per cell. Stores into
  `uns['interacting_cell_results']` and (implicitly) per-cell score matrices.
- `compute_ct_interacting_cell_scores(...)` — cell-type-**aware**, per cell per ct-pair.
  Stores per-cell dataframes in `adata.obsm['ct_interacting_cell_results_{p,np}_{gp,m}_cs_df']`
  with columns like `"CT1 - CT2: metabolite"`.
- Downstream: `compute_interaction_module_correlation(...)` correlates those per-cell scores
  against `obsm['module_scores']` (Hotspot gene modules) → which metabolite interactions
  track which expression modules.

**The exact call Foster is making** (`run_harreman.ipynb`, cell 9) is the non-ct one:
```python
harreman.tools.compute_interacting_cell_scores(
    adata, center_counts_for_np_test=False, test='both',
    restrict_significance='both', compute_significance='both',
    M=1000, seed=42, check_analytic_null=False, verbose=True)
```
`compute_significance='both'` + `M=1000` is precisely what triggers the giant permutation-null
allocation below; `test='both'` runs the parametric *and* non-parametric passes. (He also
inspects `uns['ct_interacting_cell_results']` in cell 12 — the ct twin has the same problem.)

**Why it OOMs on large data (100k–1M cells).** These functions allocate **dense tensors with
a per-cell axis**, on GPU:

- `compute_interacting_cell_scores` (non-parametric permutation) allocates
  `perm_cs_gp_a`, `perm_cs_gp_b` of shape **`(n_cells, n_gene_pairs_sig, M)`** and
  `perm_cs_m_a`, `perm_cs_m_b` of shape **`(n_cells, n_metabolites, M)`** — i.e. it stores the
  *entire permutation null for every cell*. At 100k cells × 150 pairs × M=1000 × 8 bytes ≈
  **120 GB per array** (×4). This is the crash.
- `compute_ct_interacting_cell_scores` allocates `cs_gp` of shape
  **`(n_ct_pairs, n_cells, n_gene_pairs_sig)`** (dense) plus a metabolite twin — e.g. 15 × 100k
  × 150 × 8 ≈ **180 GB** — and OOMs at the *score* step (it doesn't even implement the
  permutation test).

**Fix directions (noted, not implemented):**
1. **Don't materialize the permutation null.** The aggregate tests already do the right thing:
   they accumulate an exceedance count `x = (perm_cs > cs).sum(dim=M)` — you can accumulate
   that counter **incrementally per permutation** and never store the `(…, M)` axis. Porting
   that pattern to the per-cell path removes the `M` dimension entirely.
2. **Chunk the cell (and/or gene-pair) axis** and move results to CPU / write to disk per
   chunk; the ct loop already iterates ct-pairs, so chunk cells inside it.
3. **Lower `M`** or use the analytic Z-score (parametric) for the per-cell scores.
4. Keep dense per-cell matrices as **sparse**/`float32` and off-GPU where possible.

If we ever need per-cell metabolite scores at Xenium scale, option (1) is the surgical fix.
(Also relevant to the tractability analysis in `02_metab_integration_notes.md`.)

**Implemented (2026-07-16):** `metab_processing/Harreman/cell_communication_lowmem.py`
(renamed 2026-07-20, CU-A — was `interacting_cell_scores_lowmem.py`; the module now also holds
low-mem drop-ins for the **two aggregate** functions — see **§5c**) is a
drop-in `compute_interacting_cell_scores_lowmem` applying fix (1) — it accumulates the
exceedance counters per permutation instead of storing the `(cells, pairs, M)` arrays, and
no longer writes the raw `perm_cs_*` to `uns`. It reuses harreman's own helpers (imported
from the installed package at runtime), so numerics match the stock function for a given
seed. `run_harreman.ipynb` cell 9 now calls it (stock call kept commented). **Untested
locally** (harreman is Savio-only); validate on Savio — the file's footer has a small
diff-against-stock sanity check. Confirmed OOM site: `harreman/tools/cell_communication.py`
line ~1793 (`perm_cs_gp_a = torch.zeros((n_cells, n_gene_pairs, M), ...)`); the example run
was 112,551 cells × 136 sig pairs × M=1000 → ~114 GiB *per array* (×4). Only the ct twin
(`compute_ct_interacting_cell_scores`) is still un-patched — it needs cell-axis chunking as
well since its `cs_gp` is dense `(n_ct_pairs, n_cells, n_gene_pairs)`.

### 5a. Neighborhood scores in the wrapper (`nbhd_scores.py`, 2026-07-18) ✅ ran on Savio
harreman has **no table-builder for the per-cell scores** — `select_significant_interactions`
reduces tests 7/8 into `cell_com_df_*`, but `compute_interacting_cell_scores` just leaves raw
`(n_cells, n_interactions)` matrices in `uns['interacting_cell_results']`. The only shipped
consumers are `compute_interaction_module_correlation` and the two `plots.plot_*` functions.

`HarremanRunner(data_path, compute_nbhd_scores=True)` (default on; `False` = previous behavior
exactly) computes them once in `run_cell_independent` — they are cell-type-**independent**, so
every tier is a groupby over the same matrices, no recompute. `save_harreman_outputs` writes
`<Tier>/[nbhd_scores][summary_{m,gp}].csv`: one row per (cell type, metabolite | gene pair)
with `n_cells, frac_sig, mean_cs, mean_cs_sig, mean_neg_log10_pval, log2_enrichment`.
Significance = harreman's own `selected` rule (`FDR < alpha AND cs > 0`).

⚠️ **This is not the ct statistic.** It buckets each cell's score by that cell's *own* label;
`ct_ccc_results` restricts weights to a CT1–CT2 interface (§3). Read it as "which cell types
sit in high-scoring neighborhoods", never as an interface or a direction. It is a cheap
complement to `compute_ct_interacting_cell_scores` (still OOM, still no permutation test).
Also: `log2_enrichment` is unstable for small labels — melanoma Tier3 top hits sat in a
239-cell type with 3 significant cells. Filter on `n_cells`/`n_cells_sig` before ranking on it.
The stored FDRs are BH over the flattened cells × interactions matrix, so they are conservative.

**The `setdefault` trap (fixed).** `compute_gene_pairs` saves with `adata.uns.setdefault(...)`,
which only writes when a key is *absent*. So a second tier in the same session kept the **first
tier's** `cell_type_pairs` / `gene_pairs_per_ct_pair` and died with `KeyError: <tier-1 label>`
in `create_weights_ct_pairs`. `run_cell_aware` now pops both keys first. Everything else on the
ct path is written by direct assignment and refreshes on its own. With that fix, running all
tiers is a plain loop — no orchestration needed — which is what `run_full_harr.ipynb` does.
(`gene_pairs` / `gene_pairs_per_metabolite` are `setdefault` too, and are frozen by the first
`ct_specific=False` call, but that is harmless here — see §3 item 3.)

### 5b. `run_full_harr.ipynb` (2026-07-18)
Loops datasets under the Xenium data dir → runs harreman on whichever of Tier1/2/3 are in
`adata.obs` → writes `summary/{metabolite,gene_pair}_summary.csv` and the SpaceTravLR
`metabolite_selection.yaml`. Per-dataset `try/except` so a failure (no adata, no annotations)
skips and retries next run. A `.dataset_name` marker is written to `easy_download/` **only on
full success** and is what the skip check reads.

### 5c. Aggregate CCC memory fix — gene-pair chunking (CU-A–D, 2026-07-20) ⏳ validate on Savio
The OOM that hit when **running `HarremanRunner` itself** (not the notebook per-cell scores of
§5) is in the **two aggregate CCC functions**, and it is a *different* problem from §5's
per-cell `(cells, pairs, M)` blowup. Key facts (verified against the harreman source):

- **`compute_cell_communication` / `compute_ct_cell_communication` do NOT have the per-cell×M
  OOM.** They reduce the score over the cell axis (`.sum(0)`) *before* building the permutation
  null, so their `perm_cs_*` arrays are `(n_gp, M)` / `(n_ct_pairs, n_gp, M)` — megabytes, not
  the 100+ GB of the per-cell functions. **Lowering `M` does not help them.** Their real cost is
  the dense **`(n_cells × n_gene_pairs)` matmul intermediates** (`counts_1/2`, `WX2t`, `WtX2t`,
  and the parametric `WX1t/WtX1t`) — a few GB *each* at ~1M cells, several live at once → OOM.
- **Fix = gene-pair chunking.** The score `(counts_1.T · WX2t).sum(0)` sums over *cells*; slicing
  the *gene-pair* columns into blocks is **provably bit-identical** (each pair still sums over all
  cells; `sparse.mm` computes each column independently). Chunking `counts_1/2` rows is also safe
  because harreman's `standardize_counts` → `danb_model_torch` is strictly per-gene-row (uses only
  that row + the global `num_umi`). So only `(chunk × n_cells)` ever exists; peak memory drops from
  ~20 GB to ~1–2 GB at 1M cells. The permutation loop is restructured **outer-chunk / inner-M with
  `torch.manual_seed(seed)` re-issued per chunk**, which replays the identical `idx` sequence so the
  small `(n_gp, M)` null is reconstructed exactly.

**Two real stock quirks we had to reproduce for bit-identity** (both look like bugs; flag to the
harreman authors): (1) the cell-indep pval divide casts the exceedance count to **float32**
(`(x+1).float()/(M+1)`); (2) the **ct** function allocates `cs_gp = torch.zeros((n_ct_pairs, …))`
with **no dtype → float32**, silently truncating `cs`/`EG2`/the metabolite sum while everything
else is float64. The drop-ins replicate both.

**Where it lives:** `metab_processing/Harreman/cell_communication_lowmem.py` now holds **all three**
drop-ins (annotated `# STOCK:` diff style so a harreman maintainer can see exactly what changed):
`compute_interacting_cell_scores_lowmem` (§5, per-cell, CU-A), `compute_cell_communication_lowmem`
(CU-B), `compute_ct_cell_communication_lowmem` (CU-C). `harreman_funcs.py` calls all three
(CU-D): the two aggregates directly, the per-cell one via `nbhd_scores.py`. Chunk size is
`gene_pair_chunk_size` on `HarremanRunner` — `None` (default) auto-sizes to a ~50M-element budget
(`max(1, 50_000_000 // n_cells)`); identical output for any chunk size.

**Testing:** exhaustive local equivalence vs **vendored** stock harreman (byte-identical to the
`../Harreman` clone), in `tests/test_cell_communication_{,agg_,ct_}lowmem.py` + `test_harreman_funcs_wiring.py`
(run in `spacetravlr_env`; no real harreman needed — `tests/fixtures/fake_harreman/`). `cs` and the
**entire non-parametric path** (`pval`/`FDR`/`perm_cs` — what `select_significant_interactions`
gates on) are **exactly bit-identical** across all chunk sizes and 100s of seeds; parametric `Z`
wobbles ≤ ~2 ULP (float64 reduction-order noise from chunking, off the production-gating path).

⚠️ **GPU caveat + Savio gate.** All local proof is on **CPU**. On CUDA, reduction kernels can
reorder sums by tensor width, so even `cs`/`perm_cs` *could* pick up ULP drift (a perm value within
a ULP of the observed score could flip one integer exceedance — measure-zero, but possible). Run
`metab_processing/Harreman/validate_lowmem_savio.py --adata … --cell-type-col … --chunk-size 8`
on Savio (real harreman + GPU) as the final gate before a production run; it prints max per-key diff
for all three functions. **Tractability:** chunking adds ~`n_chunks×` more `sparse.mm` launches (total
FLOPs unchanged) — budget wall-clock accordingly, but stock OOMs outright at this scale.

---

## 6. Quick "how do I…" index
- *Is metabolite M exchanged at all?* → `[ccc_results][cell_com_df_m].csv`, `selected`/`FDR_np`.
- *Between which cell types (this annotation)?* → `Tier*/[ct_ccc_results][cell_com_df_m].csv`,
  read pairs as **undirected** interfaces (§3).
- *Which transporter genes carry it there?* → `Tier*/[ct_ccc_results][cell_com_df_gp_sig].csv`
  (+ `gp_per_metabolite` in the network JSON for the metabolite↔pair map).
- *One tidy view of all of the above* → `metab_processing/Harreman/harreman_summary.py`.
- *Per-cell "who is talking"* → the interacting-cell-score funcs (§5) — **OOM risk**.
- *Which cell types carry a metabolite's neighborhood score?* → `<Tier>/[nbhd_scores][summary_m].csv`
  (§5a) — mind the "not the ct statistic" warning.
- *Run every dataset end to end* → `metab_processing/Harreman/run_full_harr.ipynb` (§5b).
